# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 11:08:34 2024

@author: grover.laporte
"""

import matplotlib.pyplot as plt
import altair as alt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
#from scipy.stats.mstats import linregress
import scipy.stats as sps
import seaborn as sns
plt.style.use('ggplot')
from itertools import combinations
import streamlit as st
from sklearn.neighbors import KernelDensity

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
## Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor,
                              GradientBoostingRegressor,
                              AdaBoostRegressor)
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import partial_dependence



def comb(stuff):
    ans = []
    for L in range(len(stuff) + 1):
        for subset in combinations(stuff, L):
            ans.append(subset)
    return ans

from sklearn.base import BaseEstimator, TransformerMixin

class CustomONeHotEncoder(BaseEstimator,TransformerMixin):
    def __init__(self,**kwargs):
        self.feature_names = []
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        result = pd.get_dummies(X)
        self.feature_names = result.columns
        return result


class Regression(object):
    counter_ = 0
    def __init__(self,df,response):
        ## df - original data passed to object
        self.df = df.copy()
        self.response_idx = (list(df.columns)).index(response)
        self.response = response
        self.idx = 0
        self.dd = {i:col for i,col in enumerate(df.columns)}
        self.cols = list(df.columns)
        cols = [str(cols).replace('-','_') for cols in self.cols]
        self.cols = cols
        self.df.columns = cols
        self.var = {}
        self.kmeans = {}
        for col in self.cols:
            self.var[col]={}
            tot_uniq = self.df[col].nunique()
            if tot_uniq <= 10:
                self.var[col]['type']='cat'
                self.var[col]['unique']=tot_uniq
            else:
                self.var[col]['type']='num'
                self.var[col]['unique']=tot_uniq
        ## object.analysis - dictionary of analysis attempted
        self.analysis = {}
        self.analysis['All']={}
        self.analysis['All']['cols'] = self.cols
        self.analysis['All']['name'] = 'All'
        self.analysis['All']['model']= LinearRegression()
        self.analysis_cols = []
        self.analysis_selected = 'All'
        
        ## For all of the models that we will use, have a starting value
        ## for the parameters that will be used in grid search
        self.models={}
        self.models['params']={}
        self.models['params']['n_estimators']=[3,5,10]
        self.models['params']['learning_rate']=[0.1,0.2,0.5]
        self.models['params']['max_features']=[2,4,6,8]
        
        
    def __call__(self,idx):
        """
        call the self(idx) where idx is a number returns the name of column
        """
        return self.dd[idx]
    
    def __getitem__(self,cols):
        """
        self[slice] - self[slice] returns the sliced dataframe
        """
        try:
            return self.df.iloc[:,cols]
        except:
            ### idx is a boolean mask
            return(self.df[cols])
        
    def __repr__(self):
        res = ''
        for i,col in enumerate(self.cols):
            res+=f"{i:4} - {col:40}: {self.var[col]['type']}\n\n"
        return res
    
    def __str__(self):
        res = ''
        for i,col in enumerate(self.cols):
            res+=f"{i:4} - {col:40}: {self.var[col]['type']}\n\n"
        return res
    
    def col_name_idx(self,col):
        """
        Allows a user to use the index of the column or the name of the column
            to reference it. Used in functions to follow to allow name or index
            calls for data analysis.
        """
        if isinstance(col,int):
            col_str = self.cols[col]
            col_idx = col
        if isinstance(col,str):
            col_idx = self.cols.index(col)
            col_str = col
        return col_str,col_idx
    
    
    ######## Streamlit Common Interactions ##################################
    
    def select_col(self,name):
        col = st.selectbox("Select a column: "+str(name),
                     options = self.cols,
                     index = 0,
                     key = 'select_col'+str(name))
        return col
    
    def select_radio(self,name,opts):
        type_ = st.radio("Choose one of the options below.",
                         options = opts,
                         index = 0,
                         key="select_radio"+str(name))
        return type_ 
    
    def select_cat_col(self,name):
        cols = [[col, self.var[col]['unique']] for col in self.cols 
                if self.var[col]['type']=='cat']
        col = st.selectbox("Select a categorical column: "+str(name),
                           options = cols,
                           index = 0,
                           key = "cat_col_"+str(name))
        return col[0]
    
    def select_num_col(self,name):
        cols = [[col, self.df[col].min(),self.df[col].max()] for col in self.cols 
                if self.var[col]['type']=='num']
        col = st.selectbox("Select a numerical column: "+str(name),
                           options=cols,
                           index=0,
                           key = "num_col_"+str(name))
        return col[0]
    
########## Delete columns #####################################################
    def adjust_data(self):
        def delete_column():
            self.df = self.df.drop(col_str,axis=1)
            self.__init__(self.df,self.response)
        col = self.select_col("delete")
        col_str,col_idx = self.col_name_idx(col)
        st.button("Delete",key="delete_btn",on_click=delete_column)
        
########## Add columns ########################################################        
    def add_column(self,col_name,col_data):
        self.df[col_name]=col_data
        self.__init__(self.df,self.response)

################### Files #####################################################    
    
    def set_response(self):
        def change_response():
            self.response = st.session_state.resp_change
            self.response_idx = self.cols.index(self.response)
            
        st.selectbox("Select a column as the response variable.",
                     options = self.cols,
                     index=self.response_idx,
                     on_change = change_response,
                     key = "resp_change")
        
################### Variables ################################################# 
    def create_stats(self,col):
        """
        creates the statistics and kde for numerical variable types and
            bar plot for categorical variables.
        """
        col_str,col_idx = self.col_name_idx(col)
        stats = self.df[col].describe()
        x = self.df[[col]].values
        max_ = x.max();min_=x.min()
        range_ = max_- min_
        bw = range_/25
        fig,ax = plt.subplots(figsize=(8,3))
        if self.var[col]['type']=='num':
            kde_sk = KernelDensity(bandwidth = bw,kernel='gaussian')
            kde_sk.fit(x)
            eval_pts = np.linspace(min_,max_,100)
            y = np.exp(kde_sk.score_samples(eval_pts.reshape(-1,1)))
            ax.plot(eval_pts,y)
            ax.fill_between(eval_pts,y1=y,y2=0,alpha=0.6,color='cadetblue')
            ax.set_title(col_str)
            ax.set_ylabel("Relative Frequency")
        else:
            dd=self.df.copy()
            dd['tot']=1
            dd = pd.pivot_table(dd,index = col_str,aggfunc={'tot':'count'})
            dd.plot.bar(ax=ax,rot=90,alpha=0.6)
            for p in ax.patches:
                ax.annotate(str(p.get_height()),
                            (p.get_x(),p.get_height()*1.005))
            ax.set_title(col_str)
            ax.set_ylabel("Count")
            ax.set_xlabel("")
        return stats,fig
        
################### Variable Type #############################################

    def set_var_type(self):
        """
        Works with the create_stats method to produce the Variable Type
            display for the user. The user can change the type which will
            change the type of graph displayed.
        """
        def radio_change():
            col = st.session_state.select_colvar_type
            var = st.session_state.radio_change
            self.var[col]['type']=var
        def color_coding(row):
            i = self.cols.index(col)
            return (['background-color:red']*len(row) 
                    if row.idx==i 
                    else ['background-color:green']*len(row))
        
        col = self.select_col("var_type")
        opts = ["cat","num"]
        #idx = opts.index(self.var_type[col])
        idx = opts.index(self.var[col]['type'])
        st.radio("Variable type",
                 options = opts,
                 index = idx,
                 on_change=radio_change,
                 key = "radio_change")
        dd = pd.DataFrame(self.var,index = ["type","unique"]).T
        dd['idx']=np.arange(len(dd))
        col1,col2 = st.columns(2)
        
        with col1:
            st.write("Selected Column")
            stats,fig = self.create_stats(col)
            st.write(stats)
        with col2:
            st.write("All Columns")
            dd = dd.style.apply(color_coding,axis=1)
            st.dataframe(dd)
        
        st.pyplot(fig)

###### Counts #################################################################

    def hist(self,col):
        """
        Produces two different distributions - evenly distributed in the column
            selected and quantile distributed.
        """
        col_str,col_idx = self.col_name_idx(col)
        resp_str,resp_idx = self.response,self.response_idx
        dd = self[[col_idx,resp_idx]]
        hist_ = np.histogram(dd[col_str])
        num = hist_[0]
        bins = hist_[1]
        index_=[]
        value_ = []
        for i,n in enumerate(num):
            lower = bins[i];upper = bins[i+1]
            index_.append((round(lower,2),round(upper,2)))
            m=dd[(dd[col_str]>=lower) & (dd[col_str]<=upper)][resp_str].mean()
            value_.append([n,m])
        ans1=pd.DataFrame(value_,index = index_,columns=["count","mean"])
        
        quants=np.quantile(dd[col_str],[0,0.1,.2,.3,.4,.5,.6,.7,.8,.9,1])
        hist_ = np.histogram(dd[col_str],quants)
        num = hist_[0]
        bins = hist_[1]
        index_=[]
        value_ = []
        for i,n in enumerate(num):
            lower = bins[i];upper = bins[i+1]
            index_.append((round(lower,2),round(upper,2)))
            st.write()
            m=dd[(dd[col_str]>=lower) & (dd[col_str]<=upper)][resp_str].mean()
            value_.append([n,m])
        ans2=pd.DataFrame(value_,index = index_,columns=["count","mean"])
        return ans1,ans2
    
    def val_counts(self,col):
        """
        Simple pivot with categorical columns as the index and count/mean of
            the response variable inside the pivot table.
        """
        col_str,col_idx = self.col_name_idx(col)
        resp_str,resp_idx = self.response,self.response_idx
        dd = self[[col_idx,resp_idx]]

        column_order = [col_str,resp_str]
        ans = dd.pivot_table(index = col_str,aggfunc={col_str:'count',resp_str:'mean'})
        ans=ans.reindex(column_order,axis=1)

        return ans

    def counts(self):

        col = self.select_cat_col("categorical_counts")
        col_str,col_idx = self.col_name_idx(col)
        
        try:
            st.write(self.val_counts(col_str))
        except:
            st.write("Check the response variable.")
       
        col2 = self.select_num_col("numerical_counts")
        col_str2,col_idx2 = self.col_name_idx(col2)
        ans1,ans2 = self.hist(col_str2)
        c1, c2 = st.columns(2)
        with c1:
            st.write("Even Distribution")
            st.write(ans1)
        with c2:
            st.write("Quantile Distribution")
            st.write(ans2)
       
############### Correlation ###################################################

    def correlation(self):
        """
        Given a column and value, calculate the correlation between that column and the
            response variable for both halves of the interval divided by the val.
            var_type
            
        """
        def color_coding(row):
            
            return (['background-color:red']*len(row) 
                    if row.idx==col_idx 
                    else ['background-color:white']*len(row))
        
        
        resp_str,resp_idx = self.response,self.response_idx
        ######### Correlation - All columns ######################
        col_str,col_idx = self.col_name_idx(st.session_state['num_col_corr'][0])
        corr_matrix = self.df.corr()
        corr = corr_matrix[[resp_str]]
        corr['idx']=np.arange(len(corr))
        corr = corr.sort_values(resp_str,ascending=False)
        corr = corr.style.apply(color_coding,axis=1)
        st.write(corr)
        
        st.divider()
        st.subheader("Nonlinear Correlation")
        col = self.select_num_col("corr")
        col_str,col_idx = self.col_name_idx(col)
        
        try:
            col1,col2 = st.columns(2)
            with col1:
                dd = self[[col_idx,resp_idx]]
                dd[col_str+'^2']=dd[col_str]**2
                dd[col_str+'^3']=dd[col_str]**3
                dd['log('+col_str+')']=np.log(dd[col_str])
                dd['sqrt('+col_str+')']=np.sqrt(dd[col_str])
                overall_corr = dd.corr()[resp_str].sort_values(ascending=False)
                st.write(overall_corr)
            with col2:
                nonlinear = ['$x^2$','$x^3$','$\log{(x)}$','$\sqrt{x}$']
                chkbx = []
                for i,txt in enumerate(nonlinear):
                    chkbx.append(st.checkbox(str(txt)))
                btn_add_cols = st.button("Add checked columns")
                if btn_add_cols:
                    cols=[]
                    df_cols = []
                    if chkbx[0]:
                        cols.append(col_str+'^2')
                        df_cols.append(col_str+'_2')
                    if chkbx[1]:
                        cols.append(col_str+'^3')
                        df_cols.append(col_str+'_3')
                    if chkbx[2]:
                        cols.append('log('+col_str+')')
                        df_cols.append('log_'+col_str)
                    if chkbx[3]:
                        cols.append('sqrt('+col_str+')')
                        df_cols.append('sqrt_'+col_str)
                    self.add_column(df_cols, dd[cols].values)
        except:
            st.write("Select another numerical column")
        return None

############ Interaction Analysis #############################################

    def interaction(self):
        ## needs statsmodels.formula.api
        y_str,y_idx = self.col_name_idx(self.response_idx)
        cols = [col for col in self.cols 
                if self.var[col]['type']=='num']
        cols_ = st.multiselect("Choose two or three columns",
                              options=cols,
                              default=cols[0:2],
                              max_selections = 3,
                              )
        if len(cols_)>0:
            dd = self.df[cols_].copy()
            new_cols = comb(cols_)
            cbx = []
            for cols in new_cols:
                if len(cols)>1:
                    col_name = ''.join(str(x)+'_' for x in cols)[:-1]
                    val = np.ones(len(dd))
                    for col in cols:
                        val*=dd[col]
                    dd[col_name]=val
                    cbx.append(col_name)
            formula_ = ''.join(str(x)+'+' for x in list(dd.columns))[:-1]
            dd[y_str] = self.df[y_str]
            result = smf.ols(formula=y_str+'~'+formula_,data=dd).fit()
            st.write(result.summary())
            st.divider()
            checkbox = []
            col1,col2 = st.columns(2)
            with col1:
                for i,col in enumerate(cbx):
                    checkbox.append(st.checkbox(str(col),value=False))
            with col2:
                btn_create=st.button("Create Columns")
            
            if btn_create:
                cols_idx = [cbx[i] for i,val in enumerate(checkbox) if val]
                for col in cols_idx:
                    self.add_column(col, dd[col].values)
                st.write(cols_idx)

####### Graph Multicollinearity ###############################################

    def multicollinearity(self):
        cols = [col for col in self.cols 
                if self.var[col]['type']=='num']
        cols_ = st.multiselect("Choose columns to check multicollinearity:",
                              options=cols,
                              default=cols[0:2],
                              key='ms_multicol'
                              )
        if len(cols_)>1:
            dd = self.df[cols_].copy()
            dd = dd.assign(const=1)
            result = pd.Series([variance_inflation_factor(dd.values, i)
                               for i in range(dd.shape[1])],
                               index=dd.columns)
            result.name="Variance Inflation Factor"
            st.markdown(""":blue[Values that exceed 5 indicate a problematic
                        amount of collinearity]""")
            st.write(result)
            st.divider()
            st.subheader(""":blue[Correlation]""")
            corr = dd.corr().iloc[:-1,:-1]
            #st.write(corr)
            #st.divider()
            fig,ax = plt.subplots()
            sns.heatmap(corr,cmap='vlag',annot=True,ax=ax)
            st.pyplot(fig)
            st.divider()
            st.subheader(""":blue[Scatter Plots]""")
            fig,ax = plt.subplots()
            scatter_matrix(dd[cols_],ax=ax)
            st.pyplot(fig)
        
 ##############################################################################
 #########  Linear Regression #################################################
 ##############################################################################
    
    def pipeline(self,cols):
        num_cols = [col for col in cols if self.var[col]['type']=='num']
        cat_cols = [col for col in cols if self.var[col]['type']=='cat']

        data = self.df[cols].copy()
        
        num_pipeline = Pipeline([
            ('std_scaler',StandardScaler())
            ])
        
        full_pipeline = ColumnTransformer([
            ('num',num_pipeline,num_cols),
            ('cat',OneHotEncoder(),cat_cols)
            ])
        
        X_train = full_pipeline.fit_transform(data)
                
        names = full_pipeline.get_feature_names_out()
        
        X_train = pd.DataFrame(X_train,columns = names)
        return X_train
    

        

###### Select variables for analysis ##########################################
    
    def select_variables(self):
        def update_button():
            name = st.session_state['txt_analysis']
            self.analysis[name]={}
            self.analysis[name]['cols']=self.analysis_cols
            self.anaylsis_selected = name
        names = list(self.analysis.keys())
        name=self.analysis_selected
        new_button = st.button("New",key='new_btn')
        if new_button:
            name = st.text_input("Name of Analysis:",value='',
                                 key = 'txt_analysis',
                                 on_change=update_button)
        if name in names:
            idx = names.index(self.analysis_selected)
            name = st.selectbox("Select:",index=idx,options=names,
                                key = "sb_variables")
            
            r_button = st.radio("Select Input Method:",
                                options=["Multiple","Individual"],
                                index=0)
            
            if name in self.analysis.keys():
                self.analysis_cols = self.analysis[name]['cols']
            torf = [col in self.analysis_cols for col in self.cols]
            dd = pd.DataFrame(np.c_[self.df.columns,torf],
                              columns = ["col","include"])
            
            if r_button == "Multiple":
                cols = dd[dd["include"]]['col'].values
                if len(cols)>1:
                    _start = cols[0]
                    _stop = cols[-1]
                else:
                    _start=self.cols[0]
                    _stop = self.cols[-2]
                start, end = st.select_slider("Choose the start and stop columns to include.",
                                              options = self.cols,
                                              value = (_start,_stop),
                                              key = "start_end")
                btn_click = st.button("Update",key="update_1")
                if btn_click:
                    _,start_idx = self.col_name_idx(start)
                    _,end_idx = self.col_name_idx(end)
                    self.analysis_cols=self.cols[start_idx:end_idx]
            if r_button == "Individual":
                
                btn_click = st.button("Update",key="update_2")
                
                dd = st.data_editor(dd)
                if btn_click:
                    self.analysis_cols = list(dd[dd["include"]]['col'].values)
                    if self.response in self.analysis_cols:
                        self.analysis_cols.remove(self.response)
            
            if btn_click:
                self.analysis[name] = {}
                self.analysis[name]['name']=name
                self.analysis[name]['cols']=self.analysis_cols
                self.analysis_selected = name
            
######  Multiple Linear Regression ############################################
            
    def multiple_linear_regression(self):
        y_str,y_idx = self.col_name_idx(self.response_idx)
        
        col1, col2 = st.columns(2)
        with col1:        
            names = list(self.analysis.keys())
            idx = names.index(self.analysis_selected)
            name = st.selectbox("Analysis:",index=idx,options=names,
                                key="sb_mlr")
        with col2:
            st.markdown(" ")
            st.write(" ")
            ck_box_not_all = st.checkbox("Subset",value=False)
        
        if ck_box_not_all:
            all_cols = self.analysis[name]['cols']
            cols = st.multiselect("Select the columns to include:",
                                     options = all_cols,
                                     key = "ms_mlr_cols")
        else:
            cols = self.analysis[name]['cols']
        regress = st.button("OK",key = 'btn_mlr_ok')

        if regress or 'results' in self.analysis[name]:
            y = self.df[[self.response]].values
            X_std = self.pipeline(cols)
            self.analysis[name]['X_std']=X_std
            X_std = sm.add_constant(X_std)
            self.analysis[name]['r_cols']=cols
            self.analysis[name]['model']=LinearRegression()
            
            model = sm.OLS(y,X_std)
            results = model.fit()
            self.analysis[name]['results']=results
            st.write(results.summary())
            self.analysis_selected = name
            influence = results.get_influence()
            self.analysis[name]['std_residual']=(influence.resid_studentized_internal)
            self.analysis[name]['y_hat'] = (results.predict())

#### Residual Plots ###########################################################
    def residual_plots(self):
        y_str,y_idx = self.col_name_idx(self.response_idx)
        names = list(self.analysis.keys())
        idx = names.index(self.analysis_selected)
        name = st.selectbox("Select Analysis:",index=idx,options=names,key="sb_rp")
        if 'results' in self.analysis[name].keys():
            cols = self.analysis[name]['r_cols']
            dd = self.df[cols].copy()
            dd['y'] = self.df[y_str].copy()
            dd['std_residual'] = self.analysis[name]['std_residual']
            dd['y_hat']=self.analysis[name]['y_hat']
            source = dd
            points = alt.Chart(source,title="Standardize Residual vs Y").mark_point(
                color="blue").encode(
                    x='y_hat',
                    y='std_residual')
            st.altair_chart(points,use_container_width=True)
            
            st.divider()
            
            col=st.selectbox("Choose a column for the residual plot",
                         options = cols,
                         index = 0)
            points = alt.Chart(source,
                               title = "Standardized Residual vs "+str(col)).mark_point(
                                   color = 'salmon').encode(
                                       x=str(col),
                                       y = 'std_residual')
            st.altair_chart(points,use_container_width=True)
            
############  y vs y_hat plot #################################################
    def y_v_y_hat(self):
        y_str,y_idx = self.col_name_idx(self.response_idx)
        names = list(self.analysis.keys())
        idx = names.index(self.analysis_selected)
        name = st.selectbox("Select Analysis:",index=idx,options=names,
                            key="sb_yhat")
        if 'results' in self.analysis[name].keys():
            cols = self.analysis[name]['r_cols']
            model = self.analysis[name]['results']
            dd = self.df[cols].copy()
            y_hat = model.predict()
            y = self.df[[y_str]].copy()
            dd['y']=y
            dd['y_hat']=y_hat
            dd['residual']=dd['y']-dd['y_hat']
            dd['std_residual']=dd['residual']/dd['residual'].std()
            color = st.selectbox("Color data points:",options = dd.columns)
            
            source = dd
            points = alt.Chart(source).mark_circle().encode(
                alt.X('y'),
                alt.Y('y_hat'),
                color=color)
            
            lower = dd[['y','y_hat']].min().min()
            upper = dd[['y','y_hat']].max().max()
            line = pd.DataFrame({'y':[lower,upper],'y_hat':[lower,upper]})
            line_plot = alt.Chart(line).mark_line(color='salmon').encode(
                x='y',
                y='y_hat')
            st.altair_chart(points+line_plot,use_container_width=True)
            
    def normal_probability_plot(self):
        names = list(self.analysis.keys())
        idx = names.index(self.analysis_selected)
        name = st.selectbox("Select Analysis:",index=idx,options=names,
                            key="sb_normal_prob")
        if 'results' in self.analysis[name]:
            std_res = self.analysis[name]['std_residual']
            dd = pd.DataFrame(std_res,columns=['std_res'])
            dd=dd.dropna().sort_values('std_res')
            dd = dd.reset_index(drop=True)
            n = len(dd)
            percent = (np.arange(1,n+1)-0.5)/n
            z_percentile = sps.norm.ppf(percent,0,1)
            dd['z_percentile']=z_percentile
            source=dd
            points = alt.Chart(source).mark_circle(color='salmon').encode(
                alt.X('z_percentile'),
                alt.Y('std_res'))
            lower = z_percentile.min();upper=z_percentile.max()
            line = pd.DataFrame({'z_percentile':[lower,upper],
                                 'std_res':[lower,upper]})
            line_plot = alt.Chart(line).mark_line(color='cadetblue').encode(
                x='z_percentile',
                y='std_res')
            st.altair_chart(points+line_plot,use_container_width=True)
            

###############################################################################            
########   Cross Validation different models ##################################
###############################################################################
    def cross_validation_scores(self,model,X,y):
        scores = cross_val_score(model,X,y,
                                 scoring='neg_mean_squared_error',cv=10)
        model_scores = np.sqrt(-scores)
        return model_scores

    def cv_multiple_linear_regression(self):
        y_str,y_idx = self.col_name_idx(self.response_idx)
        names = list(self.analysis.keys())
        name = self.analysis_selected
        idx = names.index(self.analysis_selected)
        name = st.selectbox("Select Analysis:",index=idx,options=names,
                            key="sb_cv_mlr")
        
        model_name=""
        col1,col2,col3,col4,col5 = st.columns(5)
        with col1:
            btn1 = st.button("Linear Regression")
        with col2:
            btn2 = st.button("Decision Tree Regressor")
        with col3:
            btn3 = st.button("Random Forest Regressor")
        with col4:
            btn4 = st.button("Gradient Boosting Regressor")
        with col5:
            btn5 = st.button("Ada Boost Regressor")
        
        if btn1:
            model_name = "Linear Regression"
            model=LinearRegression()
            self.analysis[name]['model']=model
        if btn2:
            model_name = "Decision Tree Regressor"
            model=DecisionTreeRegressor()
            self.analysis[name]['model']=model
        if btn3:
            model_name = "Random Forest Regressor"
            model=RandomForestRegressor()
            self.analysis[name]['model']=model
        if btn4:
            gbr={"n_estimators": 500,
                 "max_depth": 5,
                 "learning_rate": 0.05}
            model_name = "Gradient Boosting Regressor"
            model=GradientBoostingRegressor(**gbr)
            self.analysis[name]['model']=model
        if btn5:
            model_name = "Ada Boost Regressor"
            dtr = DecisionTreeRegressor()
            model = AdaBoostRegressor(dtr,n_estimators = 50,
                                      learning_rate = 0.1,
                                      loss='linear')
            
        if len(model_name)>0:
            st.subheader(model_name)
            y = self.df[[y_str]]
            X_std = self.analysis[name]['X_std']
            scores = self.cross_validation_scores(model,X_std,y)
            scores = pd.DataFrame(np.c_[np.arange(len(scores)),scores],
                         columns = ['trial','cv_score'])
            st.write(scores)
            st.write("Mean:",scores['cv_score'].mean())
            st.write("Std Deviation:",scores['cv_score'].std())
            
    def change_params(self,parameter):
        st.write("Check to cut")
        cut = []
        params = self.models['params'][parameter]
        for i,p in enumerate(params):
            cut.append(st.checkbox(str(p),value=False,key="chkbx"+str(i)+str(p)))
        val = st.number_input("Add a number.",key="add_val",value=None,step=1)
        btn = st.button("Change",key='add_val_change')
        if btn:
            result = [params[i] for i,p in enumerate(cut) if not(p)]
            if val is not None:
                result.append(val)
            self.models['params'][parameter]=sorted(result)
            
    def pair_plots(self):
        y_str,y_idx = self.col_name_idx(self.response_idx)
        names = list(self.analysis.keys())
        name = self.analysis_selected
        idx = names.index(self.analysis_selected)
        name = st.selectbox("Select Analysis:",index=idx,options=names,
                            key="sb_pair_plot")
        
        cols = list(self.analysis[name]['X_std'].columns)
        col = st.multiselect("Choose column(s)",
                             options = cols,
                             default = cols[0],
                             max_selections=2)
        idx_col = [cols.index(c) for c in col]
        if len(idx_col)>0:
            model = self.analysis[name]['model']
            y = self.df[[y_str]]
            X_std = self.analysis[name]['X_std']
            model.fit(X_std,y)
            PartialDependenceDisplay.from_estimator(model,X_std,idx_col)
            fig = plt.gcf()
            st.pyplot(fig,use_container_width=True)
            PartialDependenceDisplay.from_estimator(model,X_std,idx_col,
                                                    kind='both')
            st.pyplot(plt.gcf(),use_container_width=True)
            
            if len(idx_col)==2:
                idx_col = [(idx_col[0],idx_col[1])]
                PartialDependenceDisplay.from_estimator(model,X_std,idx_col)
                st.pyplot(plt.gcf(),use_container_width=True)
                
        
        
        
        
    def random_forest_grid_search(self):
        y_str,y_idx = self.col_name_idx(self.response_idx)
        names = list(self.analysis.keys())
        name=self.analysis_selected
        idx = names.index(self.analysis_selected)
        name = st.selectbox("Select Analysis:",index=idx,options=names,
                            key="sb_cv_rfgs")
        ## Figure out a way to put in parameters ... 3 per.
        self.change_params('n_estimators')
        
        st.write(self.models['params']['n_estimators'])
        
            
            
            
            


###############################################################################
#####  Streamlit Interface with Regression Tool ###############################
###############################################################################

st.header("Regression Exploration Tool")
if 'explorer' not in st.session_state:
    st.session_state['explorer'] = None
if 'calc' not in st.session_state:
    st.session_state['calc']=True
if 'num_col_corr' not in st.session_state:
    st.session_state['num_col_corr'] = [1]

    
options = ["Files",
           "Variables",
           "Linear Regression",
           "Validation and Compare"]
select = st.sidebar.radio(label = "Select the tool:",
                      options = options,
                      key='sb_select')
if select == options[0]:
    tab1, tab2 = st.tabs(["Import Data",
                               "Set Response Variable"])
    exp = st.session_state['explorer']
    with tab1:
        st.subheader("Import csv file")
        if exp is None:
            uploaded_file = st.file_uploader("Select .csv survey file.",type='csv')
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file,index_col=0,keep_default_na=True)
                exp = Regression(df,df.columns[-1])
                st.session_state['explorer']=exp
                if 'var_change' not in st.session_state:
                    st.session_state['var_change']=exp.cols[0]
    
                st.write(df)
    with tab2:
        st.subheader("Set the response variable")
        exp = st.session_state['explorer']
        if exp is not None:
            exp.set_response()
            st.write(f"The response variable is set to {exp.response}")
            
if select == options[1]:
    tab1, tab2, tab3, tab5,tab6,tab4 = st.tabs(["Variable Type","Counts",
                                "Correlation","Interactions",
                                "Multicollinearity","Delete Column"])
    
    exp = st.session_state['explorer']
    
    ## Variable Type ##
    with tab1:
        st.subheader("Variable Type")
        if exp is not None:
            exp.set_var_type()
    ## Counts ##
    with tab2:
        st.subheader("Counts")
        if exp is not None:
            exp.counts()
    ## Correlation ##
    with tab3:
        st.subheader("Correlation")
        if exp is not None:
            exp.correlation()
        
    with tab4:
        st.subheader("Adjust Data Frame")
        if exp is not None:
            exp.adjust_data()
    
    with tab5:
        st.subheader("Interactions")
        if exp is not None:
            exp.interaction()
    
    with tab6:
        st.subheader("Multicollinearity")
        if exp is not None:
            exp.multicollinearity()
            
if select == options[2]:
    tab1,tab2,tab3,tab4,tab5 = st.tabs(["Select Columns",
                                "Multiple Linear Regression",
                                "Residual Plots",
                                "Y vs Y hat",
                                "Normal Probability Plot"])
    exp = st.session_state['explorer']
    with tab1:
        st.subheader("Analysis Selection")
        if exp is not None:
            exp.select_variables()
    with tab2:
        st.subheader("Multiple Linear Regression")
        if exp is not None and len(exp.analysis.keys())>0:
            exp.multiple_linear_regression()
    with tab3:
        st.subheader("Residual Plots")
        if exp is not None and len(exp.analysis.keys())>0:
            exp.residual_plots()
    with tab4:
        st.subheader("Y vs Y hat")
        if exp is not None and len(exp.analysis.keys())>0:
            exp.y_v_y_hat()
    with tab5:
        st.subheader("Normal Probability Plot")
        if exp is not None and len(exp.analysis.keys())>0:
            exp.normal_probability_plot()
        
        

if select == options[3]:
    tab1, tab2, tab3 = st.tabs(["Cross Validation",
                                "Partial Dependence Plots",
                                "Record Model"
                                ])
    exp = st.session_state['explorer']            
    
    ## Graphs ##
    with tab1:
        st.subheader("Cross Validation")
        if exp is not None and len(exp.analysis.keys())>0:
            exp.cv_multiple_linear_regression()
    with tab2:
        st.subheader("Random Forest Grid Search")
        if exp is not None and len(exp.analysis.keys())>0:
            exp.pair_plots()
    with tab3:
        st.subheader("Random Forest Regressor")
        if exp is not None and len(exp.analysis.keys())>0:
            exp.random_forest_grid_search()
        
        


            