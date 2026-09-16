import json, numpy as np, pandas as pd
from pathlib import Path
SHOP_NAMES=['BAKERY','PIZZA_SHOP','BRUNCH_SPOT','YARN_STORE','ICE_CREAM_SHOP','PET_CAFE','SMOOTHIE_SHOP','FARMERS_MARKET']; CODE={s:i for i,s in enumerate(SHOP_NAMES)}
class ConditionalMarketForecaster:
 def __init__(self,directory,product):
  self.product=product;self.d=Path(directory);self.cfg=json.load(open(self.d/f'{product}_final_runtime_bundle_v3.json'));z=np.load(self.d/f'{product}_final_scenario_library_v3.npz',allow_pickle=True);self.z={k:z[k] for k in z.files};self.price=pd.read_csv(self.d/f'{product}_price_map_0906_v1.csv').sort_values('inventory');self.cal=pd.read_csv(self.d/f'{product}_final_calibration_all_oof_0906.csv');self.rs=pd.read_csv(self.d/f'{product}_final_residual_support_0906.csv')
 def _gate(self,s):
  if self.product=='milk':
   if s['cow_age35']<=2:return 0 if s['cow_age_mean']<=6.20714282989502 else 1
   return 2 if s['cow_total']<=17 else 3
  shops=s['shops'];fp=lambda x:min([i+1 for i,y in enumerate(shops) if y==x],default=0);br,ba=fp('BRUNCH_SPOT'),fp('BAKERY');return (0 if ba==0 else 1) if br==0 else (2 if ba<=1 else 3)
 def _branch(self,s):
  if self.product=='milk':
   if s['cow_bonus']<=31.5:return 0 if s['cow_age_mean']<=9.20714282989502 else 1
   return 2 if s['cow_bonus']<=46.5 else 3
  shops=s['shops'];fp=lambda x:min([i+1 for i,y in enumerate(shops) if y==x],default=0);br,ba=fp('BRUNCH_SPOT'),fp('BAKERY');return (0 if ba<=1 else 1) if br==0 else (2 if s['opponent_money']<=10134 else 3)
 def _scenario_shops(self,idx,k):return [[SHOP_NAMES[c] for c in row[:k] if c>=0] for row in self.z['shop_codes'][idx]]
 def _sig(self,shops):
  if self.product=='milk':mp={'PIZZA_SHOP':'P','ICE_CREAM_SHOP':'I','SMOOTHIE_SHOP':'S'}
  else:mp={'BAKERY':'B','BRUNCH_SPOT':'R'}
  return tuple(mp.get(x,'O') for x in shops)
 def _filter(self,pool,shops,stage):
  cfg=self.cfg['reveal_filter_0906_selected'][stage];p=cfg['policy'];th=int(cfg['threshold']);k=len(shops)
  if p=='none' or k==0:return pool,'state'
  seq=self._scenario_shops(pool,k);exact=np.array([tuple(x)==tuple(shops) for x in seq]);rel=np.array([self._sig(x)==self._sig(shops) for x in seq]);a=pool[exact];b=pool[rel]
  if p=='relevant':return (b,'relevant') if len(b)>=th else (pool,'state_fallback')
  if p=='exact':return (a,'exact') if len(a)>=th else (pool,'state_fallback')
  if len(a)>=th:return a,'exact'
  if len(b)>=th:return b,'relevant'
  return pool,'state_fallback'
 def _cal(self,step):
  if step<311:
   d=max(9,min(11,(step+1)//24-1));q=self.cal[self.cal.stage==f'D{d}'];return q.iloc[0] if len(q) else self.cal.iloc[0]
  od=max(0,(step-311)/24);q=self.cal[self.cal.stage.str.startswith('D12+')].copy();i=(q.obs_days-od).abs().argmin();return q.iloc[i]
 def _price(self,x):
  xx=self.price.inventory.to_numpy(float);yy=self.price.price.to_numpy(float);a=np.asarray(x,float);v=np.interp(a,xx,yy,left=yy[0],right=yy[-1]);return v,(a<xx[0])|(a>xx[-1])
 def forecast(self,s):
  step=int(s['step']);cur=float(s['inventory']);shops=list(s.get('shops',[]));N=len(self.z['inventory_full']);reveal_ood=False;residual_ood=False
  if step<239:pool=np.arange(N);cand,level=self._filter(pool,shops,'pre');paths=self.z['inventory_full'][cand,step:].astype(float)-self.z['inventory_full'][cand,step,None]+cur;state='broad'
  elif step<311:
   day=9 if step<264 else (10 if step<288 else 11);g=self._gate(s);pool=np.where(self.z[f'gate_d{day}']==g)[0];cand,level=self._filter(pool,shops,'pre');reveal_ood='fallback' in level;paths=self.z['inventory_full'][cand,step:].astype(float)-self.z['inventory_full'][cand,step,None]+cur;state=f'G{g}'
  else:
   if step>311 and ('d12_branch' not in s or 'd12_inventory' not in s):raise ValueError('post-D12 forecast requires persisted d12_branch and d12_inventory')
   b=int(s.get('d12_branch',self._branch(s)));d12=float(s.get('d12_inventory',cur if step==311 else np.nan));
   if not np.isfinite(d12):raise ValueError('post-D12 forecast requires persisted d12_inventory')
   pool=np.where(self.z['d12_branch']==b)[0];cand,level=self._filter(pool,shops,'post');reveal_ood='fallback' in level;e=min(step-311,self.z['d12_delta'].shape[1]-1);cen=self.z[f'center_b{b}'].astype(float);base=self.z['d12_delta'][cand].astype(float)
   if e==0:delta=base[:,e:]
   else:
    rho=self.z[f'rho_b{b}'][e,e:].astype(float);robs=(cur-d12)-cen[e];R=base-cen;delta=cen[e:][None,:]+rho[None,:]*robs+(R[:,e:]-rho[None,:]*R[:,[e]]);near=self.rs.iloc[(self.rs.obs_days-(e/24)).abs().argsort()[:1]];r=near[near.branch==b]
    if len(r):residual_ood=bool(robs<r.q01.iloc[0] or robs>r.q99.iloc[0])
   paths=d12+delta;state=f'B{b}'
  qs=np.quantile(paths,[.1,.25,.5,.75,.9],axis=0);c=self._cal(step);med=qs[2].copy();qs[0]=med-c.cal_mult80*(med-qs[0]);qs[4]=med+c.cal_mult80*(qs[4]-med);qs[1]=med-c.cal_mult50*(med-qs[1]);qs[3]=med+c.cal_mult50*(qs[3]-med);pp,pood=self._price(paths);iq=qs;pq=np.stack([self._price(iq[4])[0],self._price(iq[3])[0],self._price(iq[2])[0],self._price(iq[1])[0],self._price(iq[0])[0]]);flags={'reveal_ood':reveal_ood,'residual_ood':residual_ood,'price_ood':bool(np.any(pood))};conf=1.0-0.2*sum(flags.values());return {'state':state,'conditioning_level':level,'scenario_count':len(cand),'weights':np.ones(len(cand))/len(cand),'inventory_paths':paths,'price_paths':pp,'inventory_q':iq,'price_q':pq,'ood':flags,'confidence':max(.2,conf)}
 @staticmethod
 def candidate_inventory(reference_paths,candidate_impact,baseline_impact):return reference_paths+np.asarray(candidate_impact)-np.asarray(baseline_impact)
 @staticmethod
 def scenario_value(candidate_inventory,quantities,revenue_callback,costs=None,weights=None,discount=None):
  S,H=candidate_inventory.shape;w=np.ones(S)/S if weights is None else np.asarray(weights)/np.sum(weights);d=np.ones(H) if discount is None else np.asarray(discount);c=np.zeros(H) if costs is None else np.asarray(costs);v=np.array([sum(d[h]*(revenue_callback(quantities[h],candidate_inventory[s,h])-c[h]) for h in range(H)) for s in range(S)]);return {'expected':float(w@v),'q10':float(np.quantile(v,.1)),'q50':float(np.quantile(v,.5)),'q90':float(np.quantile(v,.9))}
