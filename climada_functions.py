#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 12:06:06 2022

@author: lucaseverino
"""
## imports
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
import copy as cp
from climada.hazard import Hazard,Centroids
from climada.engine import Impact
from scipy import sparse
from timeit import default_timer as timer 
from constants import *
from functions import *


def filter_imp(imp,flt):
    imp_ae = imp.at_event
    mask = np.where(imp_ae>flt)
    imp_ae_flt = imp_ae[mask]
    aai_agg_flt = np.sum(imp_ae_flt)*imp.frequency[0]
    return aai_agg_flt

def norm_impf(impf,vmax=60,scale=1):
    '''Function that takes an impact function object and return its normalized intensity version'''
    norm_impf = cp.deepcopy(impf)
    if impf.name == 'Emanuel 2011':
        #allow for extension of the intensity
        new_intensity = np.arange(0,2*np.max(norm_impf.intensity),5)
        norm_impf = ImpfTropCyclone.from_emanuel_usa(intensity=new_intensity)
        norm_impf.intensity = scale*norm_impf.intensity/vmax
        norm_impf.paa = np.ones(norm_impf.intensity.shape)
        
    else:
        #cap values
        norm_impf.intensity = scale*norm_impf.intensity/vmax
        norm_impf.intensity = np.append(norm_impf.intensity, [10])
        norm_impf.mdd = np.append(norm_impf.mdd, [np.max(norm_impf.mdd)])
        norm_impf.paa = np.append(norm_impf.paa, [np.max(norm_impf.paa)])
    norm_impf.id=0
    norm_impf.haz_type='WS'
    
    return norm_impf

def get_imp_df(imp):
    """Function that takes a CLIMADA impact object as input and returns a pd.DataFrame with event_id event_name and imp.at_event
        as columns, sorted by decreasing impact"""
    imp_df = pd.DataFrame({"event id":imp.event_id,"event names":imp.event_name,"impacts":imp.at_event})
    imp_df = imp_df.where(imp_df.impacts>0).dropna()
    imp_df = imp_df.sort_values(by="impacts",ascending=False)
    return imp_df

def get_iday_imem(imp_df):
    """Function that takes a imp_df row from get_imp_df function as input and returns the id of the day and the of the member
        for the selected impact"""
    ev_name = imp_df["event names"]
    ev_name = ev_name[1:-1].split(",") #get rid of uncessary characters
    memid = int(ev_name[0])
    dayid = int(ev_name[1])
    return memid,dayid

def get_imp(imp_df,iday,imem):
    """Function that takes a imp_df from get_imp_df function as input and returns a pd Series corresponding to the day and member given as input"""
    evid = str((imem,iday))
    return imp_df.where(imp_df["event names"]==evid).dropna().squeeze()

def get_qt_id(imp_df,qt):
    n_ev = imp_df.shape[0]
    ev_id = int(round(n_ev*qt,1))
    return imp_df.iloc[ev_id]

def sel_reg_exp(reg_ids,ctr_list,exp):
    if 'United Kingdom' in ctr_list: #manually add UK
        reg_ids.append(826)
    if 'Kingdom of the Netherlands' in ctr_list:
        reg_ids.append(528)
    if 'Austria' in ctr_list:
        reg_ids.append(40)
    if 'Moldova' in ctr_list:
        reg_ids.append(498)
    if 'Czech Republic' in ctr_list:
        reg_ids.append(203)
    if 'Macedonia' in ctr_list:
        reg_ids.append(807)
    
    sel_exp = cp.deepcopy(exp)
    sel_exp.gdf = sel_exp.gdf.where(sel_exp.gdf['region_id'].isin(reg_ids)).dropna()
    return sel_exp

def tune_impf(impf_set,param,modname,fun_id=0,haz_type='WS',plot=False):
    impf = impf_set.get_func(haz_type=haz_type,fun_id=fun_id)
    impf_name =  impf.name
    new_impf = cp.deepcopy(impf)
    new_impf.mdd = param*new_impf.mdd
    new_impf.id = 1 # calibrated impf
    new_name = impf_name+': '+modname
    new_impf.name = new_name
    impf_set.append(new_impf)
    if plot:
        impf_set.plot()
    return impf_set

def set_centroids_v1(da,stack=True,haztype='WS',timeres="day",plot=False, printout=False):
    '''Function which takes a xarray.DataArray as an input and returns a climada.hazard object, with centroids corresponding to 
        latitude and longitude of the DataArray.'''
    
    da = da.reindex(lat=list(reversed(da.lat))) #need to invert latitude
    period = da.name
    lat = da.lat
    lon = da.lon
    
    #compute mean grid spacing (irregular model grid) and round it to 1 digit
    latreso, lonreso = get_lat_lon_res(da)
    latres = latreso.round(4)
    lonres = lonreso.round(4)
    
    #get bondaries and round to 1 digit
    min_lat=lat[0].to_numpy().round(4)
    max_lat=lat[-1].to_numpy().round(4)
    min_lon=lon[0].to_numpy().round(4)
    max_lon=lon[-1].to_numpy().round(4)    
    
    # number of points
    n_lat = len(da.lat)
    n_lon = len(da.lon)
    
    #stack first members and time resolution, and then latitutde and longitude to obtain the event matrix
    if stack:
        events = da.stack(events=("member",timeres)) #need to be of size n_ev * ncent = nmem*180 * ncent
        nmem = len(da["member"])
    else:
        events = da.rename({"day":"events"})
        nmem = 1
    evmat = events.stack(cent=("lat","lon"))
    #print('Evmat: '+str(evmat.shape))
    #nmon = len(da[res]) 
    n_ev = len(evmat["events"])
    
    #intiate Hazard object, using Centroids.from_pix_bounds
    haz = Hazard(haztype)
    left, bottom, right, top = min_lon, min_lat, max_lon, max_lat
    #scale with 1/2 res?
    xf_lat = min_lat - 0.5*latres
    xo_lon = min_lon - 0.5*lonres
    d_lat = latres
    d_lon = lonres
    #haz.centroids = Centroids.from_pnt_bounds((left, bottom, right, top), xyres) # default crs used
    #haz.centroids = Centroids.from_lat_lon(lat,lon)
    haz.centroids = Centroids.from_pix_bounds(xf_lat, xo_lon, d_lat, d_lon, n_lat, n_lon) # default crs used
    #print('Centroids shape: '+ str(haz.centroids.shape)+', size: '+str(haz.centroids.size))
    haz.intensity = sparse.csr_matrix(evmat)
    haz.units = 'm/s'
    ev_id = np.arange(n_ev, dtype=int)
    haz.event_id = ev_id
    #dayid = ["ID: "+str(ev_id[evid])+ "day: "+ str(day) for evid,day in enumerate(nmem*da.time.dt.strftime('%Y-%m-%d').to_series().to_list())]
    ev_names = events.coords["events"].to_numpy().tolist()
    haz.event_name = ev_names
    #haz.date = [721166]
    haz.orig = np.zeros(n_ev, bool)
    haz.frequency = np.ones(n_ev)/(nmem*30)
    haz.fraction = haz.intensity.copy()
    haz.fraction.data.fill(1)
    haz.centroids.set_meta_to_lat_lon()
    haz.centroids.set_geometry_points()
    haz.check()
    if printout:
        print('Lat resolution original: '+str(-latreso)+' ,rounded: '+str(-latres)+
              '\nLon resolution original: '+str(lonreso)+' ,rounded: '+str(lonres))
        print('lat: '+str(n_lat)+', lon: '+str(n_lon))
        print('Check centroids borders:', haz.centroids.total_bounds)
    if plot:
        haz.centroids.plot()
    
    return haz

def set_centroids(da,stack=True,haztype='WS',timeres="day",plot=False, printout=False):
    '''Function which takes a xarray.DataArray as an input and returns a climada.hazard object, with centroids corresponding to 
        latitude and longitude of the DataArray.'''
    
    period = da.name
    lat = da.lat
    lon = da.lon 
    
    # number of points
    n_lat = len(lat)
    n_lon = len(lon)
    
    #stack first members and time resolution, and then latitutde and longitude to obtain the event matrix
    if stack:
        events = da.stack(events=("member",timeres)) #need to be of size n_ev * ncent = nmem*180 * ncent
        nmem = len(da["member"])
    else:
        events = da.rename({"day":"events"})
        nmem = 1
    evmat = events.stack(cent=("lat","lon"))
    #print('Evmat: '+str(evmat.shape))
    n_ev = len(evmat["events"])
    
    #intiate Hazard object, using Centroids.from_pix_bounds
    haz = Hazard(haztype)
    
    #initiate lat and lon array for Centroids.from_lat_lon
    lat_ar = lat.values.repeat(len(lon))
    lon_ar = lon.values.reshape(1,n_lon).repeat(n_lat,axis=0).reshape(n_lon*n_lat)
    
    haz.centroids = Centroids.from_lat_lon(lat_ar,lon_ar)
    haz.intensity = sparse.csr_matrix(evmat)
    haz.units = 'm/s'
    ev_id = np.arange(n_ev, dtype=int)
    haz.event_id = ev_id
    ev_names = events.coords["events"].to_numpy().tolist()
    haz.event_name = ev_names
    haz.orig = np.zeros(n_ev, bool)
    haz.frequency = np.ones(n_ev)/(nmem*30)
    haz.fraction = haz.intensity.copy()
    haz.fraction.data.fill(1)
    haz.centroids.set_meta_to_lat_lon()
    haz.centroids.set_geometry_points()
    haz.check()
    if printout:
        print('Lat resolution original: '+str(-latreso)+' ,rounded: '+str(-latres)+
              '\nLon resolution original: '+str(lonreso)+' ,rounded: '+str(lonres))
        print('lat: '+str(n_lat)+', lon: '+str(n_lon))
        print('Check centroids borders:', haz.centroids.total_bounds)
    if plot:
        haz.centroids.plot()
    
    return haz

def imp_calc(modname,exp,impf_set,pathin,basename,qt=0.98,cut=1000000,if_id=0,timeres='day',pastname='historical',futname='ssp585',
             savename=None,savefiles={"haz":False,"impcsv":False,"impmat":False,"impds":False}):
    '''Function to compute impacts. Does not require already preprocessed fields. Return hazard and impact climada objects.'''  
    #read netcdf
    fn = pathin+modname+'_'+basename+".nc"
    ncdf = xr.open_dataset(fn)
    ncdfw = ncdf[[pastname,futname]]
    #apply gust factor
    gust_ds = gst_fact*ncdfw
    
    #get name of impact_func
    if_name = impf_set.get_func(haz_type=haz_type,fun_id=0).name
    
    #preprocess fields
    preprocess_func = pp_func_dic[if_name]
    gust_pp = preprocess_func(gust_ds,qt,cutarea=cut,timeres=timeres,pastname=pastname,futname=futname)
    gust_pp_past = gust_pp[pastname]
    gust_pp_fut = gust_pp[futname]
    
    #prepare hazards centroids
    haz_past = set_centroids(gust_pp_past,timeres=timeres)
    haz_fut = set_centroids(gust_pp_fut,timeres=timeres)
    
    # Exposures: rename column and assign id
    exp.gdf.rename(columns={"impf_": "impf_" + haz_type}, inplace=True)
    exp.gdf['impf_' + haz_type] = if_id
    exp.check()
    exp.gdf.head()
    #deepcopy exposure before assigning centroids
    exp_past = cp.deepcopy(exp)
    exp_fut = cp.deepcopy(exp)
    
    #assign centroids
    exp_fut.assign_centroids(haz_fut,distance='euclidian')
    exp_past.assign_centroids(haz_past,distance='euclidian')
    
    
    #compute impacts
    #past
    start_time = timer()
    imp_past = Impact()
    imp_past.calc(exp_past, impf_set, haz_past, save_mat=True) #Do not save the results geographically resolved (only aggregate values)
    time_delta_past = timer() - start_time
    print(time_delta_past)
    #future
    start_time = timer()
    imp_fut = Impact()
    imp_fut.calc(exp_fut, impf_set, haz_fut, save_mat=True)
    time_delta_fut = timer() - start_time
    print(time_delta_fut)

    ##save files
    if savename:
        savenameb = savename
    else:
        savenameb = basename
    
    savename_past = 'hist_'+modname+'_'+savenameb
    savename_fut = 'ssp585_'+modname+'_'+savenameb
    impf_namesht = impf_sht_names[if_name]
    #save hazards
    if savefiles["haz"]:
        haz_past.write_hdf5('results/hazards/'+savename_past+'.h5')
        haz_fut.write_hdf5('results/hazards/'+savename_fut+'.h5')
    #save impacts
    if savefiles["impcsv"]:
        imp_past.write_csv('results/impacts/impact csv'+impf_namesht+"_"+savename_past+'.csv')
        imp_fut.write_csv('results/impacts/impact csv'+impf_namesht+"_"+savename_fut+'.csv')
        
    if savefiles["impmat"]:
        imp_past.write_sparse_csr('results/impacts/impact matrices/'+impf_namesht+"_"+savename_past+'.npz')
        imp_fut.write_sparse_csr('results/impacts/impact matrices/'+impf_namesht+"_"+savename_fut+'.npz')
        
    if savefiles["impds"]:
        imp_past.write_sparse_csr('results/impacts/impact matrices/'+impf_namesht+"_"+savename_past+'.npz')
        imp_fut.write_sparse_csr('results/impacts/impact matrices/'+impf_namesht+"_"+savename_fut+'.npz')

    return haz_past, haz_fut, imp_past, imp_fut

def imp_calc_hist_fut(res_df,res_df2,modname,exp,impf_set,basename,pathin,pathout,if_id=0,qt=0.98,cut=1000000,gst_fact=1.67,timeres='day'
                      ,pastname='historical',futname='ssp585',savefiles={"haz":False,"impcsv":False,"impmat":False},
                     savenamehaz=None,savenameimp=None):
    '''Function to compute impacts and compute some statistic, between historical and future periode. Return dataframe res_df containing
        AAI_agg, 45yrs and 90 yrs impacts for each period, model and impact function'''
    
    #read netcdf
    fn = pathin+modname+'_'+basename+".nc"
    ncdf = xr.open_dataset(fn)
    ncdfw = ncdf[[pastname,futname]]
    #apply gust factor
    gust_ds = gst_fact*ncdfw
    
    #get name of impact_func
    if_name = impf_set.get_func(haz_type=haz_type,fun_id=0).name
    
    #preprocess fields
    preprocess_func = pp_func_dic[if_name]

    gust_pp = preprocess_func(gust_ds,qt,cutarea=cut,timeres=timeres,pastname=pastname,futname=futname)
    gust_pp_past = gust_pp[pastname]
    gust_pp_fut = gust_pp[futname]
    
    #prepare hazards centroids
    haz_past = set_centroids(gust_pp_past,timeres=timeres)
    haz_fut = set_centroids(gust_pp_fut,timeres=timeres)
    
    # Exposures: rename column and assign id
    exp.gdf.rename(columns={"impf_": "impf_" + haz_type}, inplace=True)
    exp.gdf['impf_' + haz_type] = if_id
    exp.check()
    #exp.gdf.head()
    #deepcopy exposure before assigning centroids
    exp_past = cp.deepcopy(exp)
    exp_fut = cp.deepcopy(exp)
    
    #assign centroids
    exp_fut.assign_centroids(haz_fut,distance='euclidian')
    exp_past.assign_centroids(haz_past,distance='euclidian')
    
    
    #compute impacts
    #past
    start_time = timer()
    imp_past = Impact()
    imp_past.calc(exp_past, impf_set, haz_past, save_mat=savefiles['impmat']) #Do not save the results geographically resolved (only aggregate values)
    time_delta_past = timer() - start_time
    print(time_delta_past)
    #future
    start_time = timer()
    imp_fut = Impact()
    imp_fut.calc(exp_fut, impf_set, haz_fut, save_mat=savefiles['impmat'])
    time_delta_fut = timer() - start_time
    print(time_delta_fut)
    #compute freq curves 
    imp45_past = imp_past.calc_freq_curve(return_per=45).impact 
    imp45_fut = imp_fut.calc_freq_curve(return_per=45).impact 
    imp90_past = imp_past.calc_freq_curve(return_per=90).impact 
    imp90_fut = imp_fut.calc_freq_curve(return_per=90).impact 
    #save results
    res_df.loc[modname,("AAI_agg","past",if_name)] = imp_past.aai_agg
    res_df.loc[modname,("AAI_agg","future",if_name)] = imp_fut.aai_agg
    res_df.loc[modname,("45 yr impact","past",if_name)] = imp45_past
    res_df.loc[modname,("45 yr impact","future",if_name)] = imp45_fut
    res_df.loc[modname,("90 yr impact","past",if_name)] = imp90_past
    res_df.loc[modname,("90 yr impact","future",if_name)] = imp90_fut
    
    #save results v2
    res_df2.loc[(modname,if_name),("AAI_agg","past")] = imp_past.aai_agg
    res_df2.loc[(modname,if_name),("AAI_agg","future")] = imp_fut.aai_agg
    res_df2.loc[(modname,if_name),("45 yr impact","past")] = imp45_past
    res_df2.loc[(modname,if_name),("45 yr impact","future")] = imp45_fut
    res_df2.loc[(modname,if_name),("90 yr impact","past")] = imp90_past
    res_df2.loc[(modname,if_name),("90 yr impact","future")] = imp90_fut
    
    ##save files
    proc_names = ["qt"+str(qt)[-2:],"cutarea"+str(cut),"gst1-67"]
    impf_namesht = impf_sht_names[if_name]
    pp_funcname = str(pp_func_dic[if_name]).split(" ")[1] 

    basename_proc = make_fn(proc_names,basename)
    
    #save hazards
    if savefiles["haz"]:
        if savenamehaz is None:
            savenamehaz = pp_funcname+'_'+basename_proc
            print('/!\ savenamehaz is None, saving under '+basename_proc)

        haz_past.write_hdf5(pathout+'/hazards/'+make_fn(['haz',pastname,modname],savenamehaz,filetype='.h5'))
        haz_fut.write_hdf5(pathout+'/hazards/'+make_fn(['haz',futname,modname],savenamehaz,filetype='.h5'))
    
    #save impacts
    if savenameimp is None:
            savenameimp = impf_namesht+'_'+basename_proc
            print('/!\ savenameimp is None, saving under '+savenameimp)
            
    if savefiles["impcsv"]:
        imp_past.write_csv(pathout+'/impacts/impact csv/'+make_fn(['imp',pastname,modname],savenameimp,filetype='.csv'))
        imp_fut.write_csv(pathout+'/impacts/impact csv/'+make_fn(['imp',futname,modname],savenameimp,filetype='.csv'))
        
    if savefiles["impmat"]:
        imp_past.write_csv(pathout+'/impacts/impact matrices/'+make_fn(['imp',pastname,modname],savenameimp,filetype='.npz'))
        imp_fut.write_csv(pathout+'/impacts/impact matrices/'+make_fn(['imp',futname,modname],savenameimp,filetype='.npz'))
       
    return 
