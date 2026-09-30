"""Validate actual-run identities, timing and historical-prefix invariance."""
from pathlib import Path
import sys,json
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from quant_research.pit.panel import PointInTimePanel
from quant_research.factors.ch3 import Ch3Spec,build_ch3_factors
P=ROOT/'private_outputs/ch3_csmar_20260926'
m=pd.read_csv(P/'market_panel.csv.gz',dtype={'security_id':str},low_memory=False)
f=pd.read_csv(P/'financial_panel.csv.gz',dtype={'security_id':str})
panel=PointInTimePanel.from_frames(m,f,source='CSMAR',data_version='csmar_20260926_partial_v1')
a=pd.read_csv(P/'holdings.csv.gz',dtype={'security_id':str},parse_dates=['formation_month','realization_month'])
u=pd.read_csv(P/'universe_log.csv.gz',dtype={'security_id':str},parse_dates=['formation_month','realization_month','fundamental_available_at'])
checks={}
assert not a.duplicated(['security_id','formation_month']).any();checks['unique_holdings']=True
assert (u.fundamental_available_at.dropna()<=u.loc[u.fundamental_available_at.notna(),'formation_month']).all();checks['announcement_before_formation']=True
assert (a.realization_month.dt.to_period('M').astype('int64')-a.formation_month.dt.to_period('M').astype('int64')).eq(1).all();checks['one_month_forward_returns']=True
fac=pd.read_csv(P/'factors_strict.csv',index_col='date',parse_dates=True)
legs=pd.read_csv(P/'six_portfolios_strict.csv',parse_dates=['realization_month']).pivot(index='realization_month',columns='portfolio',values='return')
smb=(legs.SV+legs.SN+legs.SG-legs.BV-legs.BN-legs.BG)/3
vmg=(legs.SV+legs.BV-legs.SG-legs.BG)/2
assert np.allclose(smb,fac.SMB,equal_nan=True) and np.allclose(vmg,fac.VMG,equal_nan=True);checks['six_leg_identities']=True
assert np.isfinite(a.me).all() and a.me.gt(0).all();checks['positive_finite_weights']=True
assert a.loc[a.ep.lt(0),'portfolio'].isin(['SG','BG']).all();checks['negative_profit_in_growth']=True
cut=pd.Timestamp('2020-12-31')
prefix=PointInTimePanel.from_frames(panel.market[panel.market.observation_date<=cut],panel.fundamentals[panel.fundamentals.available_at<=cut],source=panel.source,data_version=panel.data_version)
spec=Ch3Spec.from_dict(json.loads((P/'config.json').read_text())) if hasattr(Ch3Spec,'from_dict') else Ch3Spec(**json.loads((P/'config.json').read_text()))
from dataclasses import replace
r=build_ch3_factors(prefix,replace(spec,sample_end='2020-12-31'))
expected=fac.loc[:cut,['MKT','SMB','VMG']]
assert np.allclose(r.factor_returns[['MKT','SMB','VMG']],expected,equal_nan=True,atol=1e-12)
checks['historical_prefix_without_future_data']=True;checks['prefix_months']=len(expected)
(P/'verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
