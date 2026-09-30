"""CSMAR partial-sample CH-3 replication. Raw files remain unchanged.
Run: .venv/bin/python scripts/reproduce_ch3_csmar.py
No unavailable return is filled in the strict series. Zero-outcome illustration
is constructed from frozen holdings, never used to choose memberships.
"""
from pathlib import Path
import sys, json, hashlib, os
from dataclasses import asdict, replace
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib'))
from quant_research.pit.panel import PointInTimePanel
from quant_research.factors.ch3 import Ch3Spec, build_ch3_factors
from quant_research.repro.statistics import pricing_regression as ordinary_pricing_regression, grs_test
from scipy.stats import norm
B=ROOT/'private_data/csmar_raw'
OUT=ROOT/'private_outputs/ch3_csmar_20260926'
START='2008-01-31'; END='2025-12-31'
FILES={'market':'月个股回报率/TRD_Mnth.csv','capital':'股本变动/TRD_Capchg.xlsx',
       'company':'公司信息/CG_Co.xlsx','earnings':'非经常性收益/FI_T2.csv',
       'announcements':'年、中、季报实际公布日期/IAR_Forecdt.csv','rf':'无风险利率/TRD_Nrrate.csv'}

def pricing_regression(y, factors, hac_lags=4):
    """OLS on observed dates; HAC score products retain calendar-month gaps.

    Missing calendar scores are zero, not zero-return observations. This fixes
    lag distances only; it does NOT fix nonrandom selection of observed months.
    """
    out=ordinary_pricing_regression(y,factors,hac_lags)
    joint=pd.concat([y.rename('__y'),factors],axis=1).dropna().sort_index()
    x=np.column_stack([np.ones(len(joint)),joint[factors.columns].to_numpy(float)])
    yy=joint.__y.to_numpy(float);beta=np.linalg.lstsq(x,yy,rcond=None)[0]
    score=pd.DataFrame(x*(yy-x@beta)[:,None],index=joint.index)
    grid=pd.date_range(joint.index.min(),joint.index.max(),freq='ME')
    z=score.reindex(grid,fill_value=0).to_numpy();meat=z.T@z
    for lag in range(1,hac_lags+1):
        c=z[lag:].T@z[:-lag];meat+=(1-lag/(hac_lags+1))*(c+c.T)
    inv=np.linalg.inv(x.T@x);se=np.sqrt((inv@meat@inv)[0,0]);t=beta[0]/se
    out.update(t_hac=float(t),p_hac=float(2*norm.sf(abs(t))),ci_hac_low_pct=float((beta[0]-norm.ppf(.975)*se)*100),ci_hac_high_pct=float((beta[0]+norm.ppf(.975)*se)*100),hac_calendar_months=len(grid))
    return out

def load(name):
    p=B/FILES[name]
    d=pd.read_excel(p,dtype=str) if p.suffix=='.xlsx' else pd.read_csv(p,dtype=str)
    if 'Stkcd' in d:d=d[d.Stkcd.str.fullmatch(r'\d{6}',na=False)].copy()
    return d

def num(d,cols):
    for c in cols:d[c]=pd.to_numeric(d[c],errors='coerce')
    return d

def asof(left,right,leftdate,rightdate):
    left=left.copy();right=right.copy()
    left[leftdate]=pd.to_datetime(left[leftdate]).astype('datetime64[ns]')
    right[rightdate]=pd.to_datetime(right[rightdate]).astype('datetime64[ns]')
    return pd.merge_asof(left.sort_values(leftdate),right.sort_values(rightdate),left_on=leftdate,right_on=rightdate,by='Stkcd',direction='backward')

def prepare():
    audit={};m=load('market');audit['raw_market_rows']=len(m)
    m=m[m.Markettype.isin(['1','4','16']) & m.Trdmnt.between('2006-12','2026-06')].copy()
    num(m,['Mclsprc','Msmvttl','Msmvosd','Mretwd','Ndaytrd'])
    assert not m.duplicated(['Stkcd','Trdmnt']).any()
    m['date']=pd.to_datetime(m.Trdmnt)+pd.offsets.MonthEnd(0)
    # Sum on a full calendar grid; absent trading months count zero trading days,
    # NOT zero returns. No stock-price/return rows are fabricated here.
    days=m.pivot(index='date',columns='Stkcd',values='Ndaytrd').reindex(pd.date_range(m.date.min(),m.date.max(),freq='ME')).fillna(0).rolling(12,min_periods=12).sum()
    m=m.merge(days.stack().rename('trading_days_12m').reset_index().rename(columns={'level_0':'date'}),on=['date','Stkcd'],how='left')
    m['trading_days_12m']=m.trading_days_12m.fillna(0)
    co=load('company')[['Stkcd','ListedDate','DelistedDate']];co['ListedDate']=pd.to_datetime(co.ListedDate)
    audit['company_duplicate_codes']=int(co.duplicated('Stkcd').sum())
    m=asof(m,co,'date','ListedDate')
    m['listing_months']=(m.date.dt.year-m.ListedDate.dt.year)*12+m.date.dt.month-m.ListedDate.dt.month
    m.loc[m.ListedDate.isna(),'listing_months']=-1
    # Latest listing episode by date handles 600018; never use future relisting.
    cap=load('capital');fields=['Nshrttl','Nshrb','Nshrh','Nshroft','Nshrprf'];num(cap,fields)
    assert not cap.duplicated(['Stkcd','Shrchgdt']).any()
    m=asof(m,cap[['Stkcd','Shrchgdt']+fields],'date','Shrchgdt')
    audit['missing_cap_rows']=int(m.Nshrttl.isna().sum())
    audit['missing_share_components']={c:int(m[c].isna().sum()) for c in fields}
    # Historical total A shares are approximated by total minus explicitly listed
    # non-A classes. Missing components remain missing, not silently zero.
    m['shares_a_proxy']=m.Nshrttl-m.Nshrb-m.Nshrh-m.Nshroft-m.Nshrprf
    m['me']=m.Mclsprc*m.shares_a_proxy
    m['valuation_me']=m.Mclsprc*m.Nshrttl
    rel=(m.valuation_me/(m.Msmvttl*1000)-1).abs()
    audit['cap_vs_vendor_relative_error_median']=float(rel.median())
    audit['cap_vs_vendor_relative_error_over_1pct']=int(rel.gt(.01).sum())
    arel=(m.me/(m.Msmvttl*1000)-1).abs()
    audit['a_cap_vs_vendor_relative_error_over_1pct']=int(arel.gt(.01).sum())
    m.loc[(m.me<=0)|(m.valuation_me<=0),['me','valuation_me']]=np.nan
    rf=load('rf');num(rf,['Nrrdata']);rf=rf[rf.Nrr1.eq('NRI01')].copy();rf['date']=pd.to_datetime(rf.Clsdt)
    assert not rf.date.duplicated().any()
    # Accrue the actual one-year deposit quote each calendar day, ACT/365.
    rf['daily_log']=np.log1p(rf.Nrrdata/100)/365
    rfg=rf.groupby(rf.date.dt.to_period('M'))
    rfm=np.expm1(rfg.daily_log.sum());counts=rfg.size()
    assert all(counts.loc[p]==p.days_in_month for p in pd.period_range('2007-12','2026-06',freq='M'))
    m['rf_monthly']=m.date.dt.to_period('M').map(rfm)
    f=load('earnings')[['Stkcd','Accper','F020102']];num(f,['F020102']);a=load('announcements')[['Stkcd','Accper','Actudt']]
    assert not f.duplicated(['Stkcd','Accper']).any()
    assert not a.duplicated(['Stkcd','Accper']).any()
    f=f.merge(a,on=['Stkcd','Accper'],how='left',validate='one_to_one')
    f=f[f.Stkcd.isin(m.Stkcd.unique()) & f.Accper.le(END)].copy()
    f['available_at']=pd.to_datetime(f.Actudt,errors='coerce')
    f['observation_date']=pd.to_datetime(f.Accper,errors='coerce')
    bad=f.available_at.isna()|(f.available_at<f.observation_date)
    audit['financial_rows']=len(f);audit['financial_missing_announcement']=int(bad.sum());audit['financial_missing_profit']=int(f.F020102.isna().sum())
    f.loc[bad].to_csv(OUT/'unmatched_financials.csv',index=False)
    f=f[~bad].copy()
    # Keep missing earnings rows: new missing reports must not silently resurrect
    # a stale earlier nonmissing profit. Ordinary net profit is not in the inputs.
    f=f.rename(columns={'Stkcd':'security_id','F020102':'earnings_reported'})
    f['trade_date']=f.available_at
    m=m[m.date.between('2007-12-31',END)].copy()
    m=m.rename(columns={'Stkcd':'security_id','date':'observation_date','Mretwd':'ret_1m','Ndaytrd':'trading_days_1m'})
    for c in ['available_at','trade_date']:m[c]=m.observation_date
    for d in [m,f]:d['source']='CSMAR';d['data_version']='csmar_20260926_partial_v1'
    m=m.sort_values(['security_id','observation_date']).reset_index(drop=True)
    # Save enough raw mapped fields to make every unit and assumption auditable.
    m.to_csv(OUT/'market_panel.csv.gz',index=False);f.to_csv(OUT/'financial_panel.csv.gz',index=False)
    audit['prepared_market_rows']=len(m);audit['prepared_financial_rows']=len(f)
    audit['source_sha256']={k:hashlib.sha256((B/v).read_bytes()).hexdigest() for k,v in FILES.items()}
    panel=PointInTimePanel.from_frames(m,f,source='CSMAR',data_version='csmar_20260926_partial_v1')
    return panel,audit

def summarise(s):
    s=s.dropna()
    r=pricing_regression(s,pd.DataFrame(index=s.index))
    r.update(mean_month_pct=float(s.mean()*100),vol_annual_pct=float(s.std()*np.sqrt(12)*100),sharpe=float(s.mean()/s.std()*np.sqrt(12)))
    return r

def recombine(a,rf,zero=False):
    rows=[]
    for dt,g in a.groupby('realization_month'):
        r=g.ret_next_1m.fillna(0) if zero else g.ret_next_1m
        z=g.assign(r=r)
        legs={p: float((h.me/h.me.sum()*h.r).sum()) if h.r.notna().all() else np.nan for p,h in z.groupby('portfolio')}
        m=float((g.me/g.me.sum()*r).sum()) if r.notna().all() else np.nan
        rows.append({'date':dt,'MKT':m-rf.loc[dt], 'SMB':np.mean([legs.get(p,np.nan) for p in ['SV','SN','SG']])-np.mean([legs.get(p,np.nan) for p in ['BV','BN','BG']]),'VMG':np.mean([legs.get(p,np.nan) for p in ['SV','BV']])-np.mean([legs.get(p,np.nan) for p in ['SG','BG']])})
    return pd.DataFrame(rows).set_index('date')

def turnover_drift(previous,realized,rf_return):
    """Previous signed positions drift on a cash-collateral account, NAV=1."""
    pnl=float((previous*realized.reindex(previous.index).fillna(0)).sum())
    cash=1-float(previous.sum())
    nav=1+pnl+cash*rf_return
    if nav<=0:raise ValueError('nonpositive collateral NAV')
    return previous*(1+realized.reindex(previous.index).fillna(0))/nav

def analyze(result,panel,audit,spec):
    a=result.assignments.copy();fac=result.factor_returns[['MKT','SMB','VMG']];rf=result.diagnostics.risk_free_monthly
    fac.to_csv(OUT/'factors_strict.csv');a.to_csv(OUT/'holdings.csv.gz',index=False);result.universe_log.to_csv(OUT/'universe_log.csv.gz',index=False)
    result.portfolio_returns.to_csv(OUT/'six_portfolios_strict.csv',index=False)
    diag=result.diagnostics.drop(columns=['breakpoints','portfolio_counts']);diag.to_csv(OUT/'monthly_audit.csv')
    result.universe_log.groupby(['realization_month','exclusion_reason']).size().unstack(fill_value=0).to_csv(OUT/'exclusions.csv')
    audit['n_months']=len(fac);audit['missing_factor_months']=fac.isna().sum().to_dict();audit['below_50_width_months']=int(diag.below_min_width.sum())
    # Strict vs explicit zero-outcome illustration use IDENTICAL assignments.
    zero=recombine(a,rf,True);zero.to_csv(OUT/'factors_zero_outcome_illustration.csv')
    assert np.allclose(fac.to_numpy(),recombine(a,rf).to_numpy(),equal_nan=True)
    audit['missing_holding_outcomes']=int(a.ret_next_1m.isna().sum())
    miss=a[a.ret_next_1m.isna()];miss.to_csv(OUT/'missing_holding_outcomes.csv',index=False)
    audit['total_holding_rows']=len(a)
    keys=panel.market[['security_id','observation_date','ret_1m']].rename(columns={'observation_date':'realization_month'})
    mm=miss.merge(keys,on=['security_id','realization_month'],how='left',indicator=True)
    audit['missing_outcome_stockmonth_absent']=int(mm._merge.eq('left_only').sum())
    audit['missing_outcome_stockmonth_present']=int(mm._merge.eq('both').sum())
    risk=[]
    for dt,g in a.groupby('realization_month'):
        risk.append({'date':dt,'missing_names':int(g.ret_next_1m.isna().sum()),'missing_market_weight':float(g.loc[g.ret_next_1m.isna(),'me'].sum()/g.me.sum())})
    pd.DataFrame(risk).to_csv(OUT/'missing_outcome_weights.csv',index=False)
    audit['missing_market_weight_mean']=float(np.mean([r['missing_market_weight'] for r in risk]))
    audit['missing_market_weight_max']=float(max(r['missing_market_weight'] for r in risk))
    stresses=[]
    for dt,g in a.groupby('realization_month'):
        legw=g.me/g.groupby('portfolio').me.transform('sum')
        for k,coef in [('SMB',{'SV':1/3,'SN':1/3,'SG':1/3,'BV':-1/3,'BN':-1/3,'BG':-1/3}),('VMG',{'SV':.5,'BV':.5,'SG':-.5,'BG':-.5,'SN':0,'BN':0})]:
            unknown=(legw*g.portfolio.map(coef)).abs().where(g.ret_next_1m.isna(),0).sum()
            stresses.append({'date':dt,'factor':k,'missing_abs_notional':unknown,'zero_return':zero.loc[dt,k], 'adverse_30pct':zero.loc[dt,k]-.3*unknown,'favorable_30pct':zero.loc[dt,k]+.3*unknown})
    pd.DataFrame(stresses).to_csv(OUT/'missing_outcome_stress.csv',index=False)
    official=pd.read_csv(ROOT/'author_data/verified_20260925/ch3_monthly_official.csv',parse_dates=['date']).set_index('date')
    periods={'all':(START,END),'paper_overlap':(START,'2016-12-31'),'extension':('2017-01-31',END)}
    stats=[];align=[]
    for label,(start,end) in periods.items():
        # Calendar-aware HAC retains gaps; observed-month selection remains.
        for mode,series in [('strict',fac),('zero_illustration',zero)]:
            for k in fac:
                v=series.loc[start:end,k];stats.append({'period':label,'mode':mode,'factor':k,'calendar_missing':int(v.isna().sum()),**summarise(v)})
                pair=pd.concat([v.rename('ours'),official[k].rename('official')],axis=1).loc[start:end].dropna()
                align.append({'period':label,'mode':mode,'factor':k,'n':len(pair),'corr':pair.ours.corr(pair.official),'mean_gap_bps':(pair.ours-pair.official).mean()*10000,'rmse_bps':np.sqrt(((pair.ours-pair.official)**2).mean())*10000})
    pd.DataFrame(stats).to_csv(OUT/'factor_statistics.csv',index=False);pd.DataFrame(align).to_csv(OUT/'official_alignment.csv',index=False)
    audit['rf_vs_author_mean_abs_bps']=float((rf-official.RF_MONTHLY).abs().dropna().mean()*10000)
    # Cross-sectional IC and decile monotonicity on known next outcomes only.
    ic=[];dec=[];reversal=[]
    lag=panel.market[['security_id','observation_date','ret_1m']].rename(columns={'observation_date':'formation_month','ret_1m':'past_return'})
    a=a.merge(lag,on=['security_id','formation_month'],how='left',validate='one_to_one')
    for dt,g in a.groupby('realization_month'):
        valid=g.dropna(subset=['ret_next_1m','ep']);ic.append({'date':dt,'n':len(valid),'EP':valid.ep.corr(valid.ret_next_1m,method='spearman'),'SMALL':(-np.log(valid.me)).corr(valid.ret_next_1m,method='spearman')})
        for signal in ['ep','past_return']:
            s=g.dropna(subset=[signal]).sort_values([signal,'security_id']).copy();nq=10 if signal=='ep' else 5
            s['q']=np.minimum(np.arange(len(s))*nq//len(s)+1,nq)
            for q,h in s.groupby('q'):
                val=float((h.me/h.me.sum()*h.ret_next_1m).sum()) if h.ret_next_1m.notna().all() else np.nan
                (dec if signal=='ep' else reversal).append({'date':dt,'q':q,'return':val,'n':len(h)})
    ic=pd.DataFrame(ic).set_index('date');ic.to_csv(OUT/'rank_ic.csv')
    icstats=[]
    for k in ['EP','SMALL']:
        z=pricing_regression(ic[k],pd.DataFrame(index=ic.index))
        icstats.append({'signal':k,'n':z['n'],'mean_rank_ic':ic[k].mean(),'std_rank_ic':ic[k].std(),'icir_monthly':ic[k].mean()/ic[k].std(),'t_hac':z['t_hac'],'p_hac':z['p_hac'],'positive_month_fraction':ic[k].gt(0).mean()})
    pd.DataFrame(icstats).to_csv(OUT/'ic_statistics.csv',index=False)
    dec=pd.DataFrame(dec);dec.to_csv(OUT/'ep_deciles.csv',index=False)
    assets=pd.DataFrame(reversal).pivot(index='date',columns='q',values='return');assets.columns=[f'REV{c}' for c in assets];assets=assets.sub(rf,axis=0)
    assets.to_csv(OUT/'reversal_test_assets_excess.csv')
    regressions=[];grs={}
    common=pd.concat([assets,fac],axis=1).dropna()
    for model,cols in [('CAPM',['MKT']),('CH3',['MKT','SMB','VMG'])]:
        for col in assets:
            regressions.append({'model':model,'asset':col,**pricing_regression(common[col],common[cols])})
        try:grs[model]=grs_test(common[assets.columns],common[cols])
        except ValueError as e:grs[model]={'error':str(e)}
    pd.DataFrame(regressions).to_csv(OUT/'pricing_regressions.csv',index=False)
    audit['grs_common_complete_months']=len(common);audit['grs']=grs
    spanning=[]
    for name in ['SMB','VMG']:
        cols=[c for c in fac if c!=name];s=pd.concat([fac[name],fac[cols]],axis=1).dropna()
        spanning.append({'factor':name,**pricing_regression(s[name],s[cols])})
    pd.DataFrame(spanning).to_csv(OUT/'spanning_regressions.csv',index=False)
    # Transparent 100% cash collateral + 100% long - 100% short simulation.
    # Missing outcomes assumed zero in both drift and P&L; NOT implementable.
    back=[];old={k:pd.Series(dtype=float) for k in ['SMB','VMG']};previous_date=None
    retmap=panel.market.pivot(index='observation_date',columns='security_id',values='ret_1m')
    for dt,g in a.groupby('realization_month'):
        for k in ['SMB','VMG']:
            h=g.copy();h['leg_w']=h.me/h.groupby('portfolio').me.transform('sum')
            coef=({'SV':1/3,'SN':1/3,'SG':1/3,'BV':-1/3,'BN':-1/3,'BG':-1/3} if k=='SMB' else {'SV':.5,'BV':.5,'SG':-.5,'BG':-.5,'SN':0,'BN':0})
            target=h.set_index('security_id').leg_w*h.set_index('security_id').portfolio.map(coef)
            prev=old[k]
            drift=turnover_drift(prev,retmap.loc[previous_date],rf.loc[previous_date]) if previous_date is not None else prev
            trade=target.sub(drift,fill_value=0).abs().sum()
            assert np.isclose(target.sum(),0) and np.isclose(target.abs().sum(),2)
            back.append({'date':dt,'factor':k,'traded_notional':trade,'gross_excess':zero.loc[dt,k], 'total_no_cost':zero.loc[dt,k]+rf.loc[dt], 'net_10bps':zero.loc[dt,k]+rf.loc[dt]-.001*trade, 'net_30bps_borrow3pct':zero.loc[dt,k]+rf.loc[dt]-.003*trade-.03/12})
            old[k]=target
        previous_date=dt
    back=pd.DataFrame(back);back.to_csv(OUT/'collateral_backtest_illustration.csv',index=False)
    btstats=[]
    for k,g in back.groupby('factor'):
        for mode in ['total_no_cost','net_10bps','net_30bps_borrow3pct']:
            s=g.set_index('date')[mode];wealth=(1+s).cumprod();dd=wealth/wealth.cummax().clip(lower=1)-1
            btstats.append({'factor':k,'mode':mode,'n':len(s),'cagr_pct':(wealth.iloc[-1]**(12/len(s))-1)*100,'max_drawdown_pct':dd.min()*100,'end_wealth':wealth.iloc[-1],'average_traded_notional':g.traded_notional.mean()})
    pd.DataFrame(btstats).to_csv(OUT/'backtest_statistics.csv',index=False)
    # Simple figures, labels English for portability; Chinese explanations in MD.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,1,figsize=(10,7),sharex=True)
    for k in ['SMB','VMG']:
        z=back[back.factor.eq(k)].set_index('date');ax[0].plot(z.index,(1+z.net_10bps).cumprod(),label=k+' cash collateral, 10bps')
    ax[0].set_yscale('log');ax[0].legend();ax[0].set_title('Illustration only: missing outcomes = 0, unconstrained long/short')
    ax[1].plot(pd.to_datetime([r['date'] for r in risk]),[r['missing_market_weight']*100 for r in risk]);ax[1].set_ylabel('Unknown-return weight (%)');fig.tight_layout();fig.savefig(OUT/'backtest_illustration.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for ax,k in zip(axes,fac):
        pair=pd.concat([fac[k].rename('ours'),official[k].rename('author')],axis=1).dropna();ax.scatter(pair.author*100,pair.ours*100,s=8,alpha=.5);ax.set_title(k);ax.set_xlabel('Author (%)');ax.set_ylabel('Replication (%)')
    fig.tight_layout();fig.savefig(OUT/'official_alignment.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3));ic[['EP','SMALL']].rolling(12).mean().plot(ax=ax);ax.axhline(0,color='gray',linewidth=.7);ax.set_title('12-month average Rank IC (pairwise observed outcomes)');fig.tight_layout();fig.savefig(OUT/'rank_ic.png',dpi=160);plt.close(fig)
    # Exact formation trace, chosen beforehand rather than based on best fit.
    trace=result.universe_log.query("security_id == '000001'").copy();trace.to_csv(OUT/'trace_000001.csv',index=False)
    return fac,zero

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    print('Preparing licensed panel',flush=True);panel,audit=prepare()
    spec=Ch3Spec(experiment_id='EXP-CH3-CSMAR-20260926',strategy_name='csmar_partial_strict',data_version=panel.data_version,sample_start=START,sample_end=END,earnings_field='earnings_reported',valuation_cap_field='valuation_me',book_to_market_control=False,risk_free_field='rf_monthly',risk_free_is_monthly_decimal=True,require_risk_free=True,require_full_sample=True)
    (OUT/'config.json').write_text(json.dumps(asdict(spec),ensure_ascii=False,indent=2))
    print('Constructing strict CH-3',flush=True);result=build_ch3_factors(panel,spec)
    expected=pd.date_range(START,END,freq='ME')
    assert result.factor_returns.index.equals(expected), 'An entire formation month is empty; do not compress the return calendar'
    print('Analyzing factors and explicit missing-outcome illustration',flush=True);fac,zero=analyze(result,panel,audit,spec)
    sensitivity=[]
    for label,ps,sp in [('no_smallcap_cut',panel,replace(spec,exclude_smallest_fraction=0)),('announcement_plus_30d',PointInTimePanel.from_frames(panel.market,panel.fundamentals.assign(available_at=panel.fundamentals.available_at+pd.Timedelta(days=30),trade_date=panel.fundamentals.trade_date+pd.Timedelta(days=30)),source=panel.source,data_version=panel.data_version),spec),('vendor_market_cap',PointInTimePanel.from_frames(panel.market.assign(me=panel.market.Msmvttl*1000),panel.fundamentals,source=panel.source,data_version=panel.data_version),spec)]:
        print('Sensitivity',label,flush=True);r=build_ch3_factors(ps,sp);v=recombine(r.assignments,r.diagnostics.risk_free_monthly,True);v.to_csv(OUT/f'sensitivity_{label}.csv')
        for k in ['MKT','SMB','VMG']:sensitivity.append({'variant':label,'factor':k,'comparison':'zero_outcome_illustration',**summarise(v[k]),'corr_baseline':v[k].corr(zero[k])})
    pd.DataFrame(sensitivity).to_csv(OUT/'sensitivity_statistics.csv',index=False)
    hac=[]
    for k in zero:
        for lag in [0,4,6,12]:hac.append({'factor':k,'mode':'zero_illustration',**pricing_regression(zero[k],pd.DataFrame(index=zero.index),lag)})
    pd.DataFrame(hac).to_csv(OUT/'hac_lag_sensitivity.csv',index=False)
    audit['versions']={'python':sys.version,'pandas':pd.__version__,'numpy':np.__version__}
    audit['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    audit['engine_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/quant_research/factors/ch3.py',ROOT/'src/quant_research/pit/panel.py',ROOT/'src/quant_research/repro/statistics.py']}
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=str))
    print(json.dumps(audit,ensure_ascii=False,indent=2,default=str),flush=True)

if __name__=='__main__':main()
