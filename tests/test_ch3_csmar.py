"""Tests for the real-data adapter's time and accounting conventions."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from quant_research.repro.statistics import pricing_regression as reference_regression

spec=importlib.util.spec_from_file_location('csmar_runner',Path(__file__).resolve().parents[1]/'scripts/reproduce_ch3_csmar.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def test_asof_never_uses_future_share_event():
    l=pd.DataFrame({'Stkcd':['000001']*3,'date':['2020-01-31','2020-02-29','2020-03-31']})
    r=pd.DataFrame({'Stkcd':['000001']*2,'event':['2020-02-10','2020-04-01'],'shares':[100,900]})
    out=mod.asof(l,r,'date','event')
    assert np.isnan(out.shares.iloc[0])
    assert out.shares.iloc[1:].tolist()==[100,100]

def test_relisting_uses_then_current_episode():
    l=pd.DataFrame({'Stkcd':['600018']*2,'date':['2005-12-31','2006-11-30']})
    r=pd.DataFrame({'Stkcd':['600018']*2,'listing':['2000-07-19','2006-10-26']})
    out=mod.asof(l,r,'date','listing')
    assert out.listing.dt.strftime('%Y-%m-%d').tolist()==['2000-07-19','2006-10-26']

def test_calendar_hac_matches_reference_for_complete_calendar():
    rng=np.random.default_rng(412);ix=pd.date_range('2010-01-31',periods=100,freq='ME')
    f=pd.DataFrame({'MKT':rng.normal(size=100)},index=ix);y=pd.Series(rng.normal(size=100),index=ix)+.3*f.MKT
    a=mod.pricing_regression(y,f);b=reference_regression(y,f)
    for k in ['alpha_pct','t_hac','p_hac','ci_hac_low_pct','ci_hac_high_pct']:assert np.isclose(a[k],b[k])

def test_calendar_hac_does_not_compress_missing_months():
    ix=pd.date_range('2010-01-31',periods=60,freq='2ME');y=pd.Series(np.sin(np.arange(60)/4)+.1,index=ix);f=pd.DataFrame(index=ix)
    a=mod.pricing_regression(y,f);b=reference_regression(y,f)
    assert a['hac_calendar_months']==119
    assert not np.isclose(a['t_hac'],b['t_hac'])

def test_signed_holdings_cash_collateral_drift():
    w=pd.Series({'long':1.,'short':-1.});r=pd.Series({'long':.1,'short':-.1})
    actual=mod.turnover_drift(w,r,.001)
    assert np.isclose(actual['long'],1.1/1.201)
    assert np.isclose(actual['short'],-.9/1.201)

def test_missing_outcomes_never_renormalize_strict_portfolio():
    dt=pd.Timestamp('2020-02-29');rows=[]
    for p in ['SV','SN','SG','BV','BN','BG']:
        rows.append({'realization_month':dt,'portfolio':p,'me':100.,'ret_next_1m':.1})
    rows[0]['ret_next_1m']=np.nan
    a=pd.DataFrame(rows);rf=pd.Series({dt:0.})
    strict=mod.recombine(a,rf);demo=mod.recombine(a,rf,True)
    assert strict.loc[dt].isna().all()
    assert np.isclose(demo.loc[dt,'MKT'],.5/6)
    assert np.isclose(demo.loc[dt,'VMG'],-.05)
