"""Every figure declares its evidence source. Missing data is shown as missing."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import read_jsonl


def _save(fig,path,source):
    fig.text(.99,.01,'SYNTHETIC — SOFTWARE VALIDATION ONLY' if source=='synthetic' else 'MEASURED — see sample sizes and protocol',ha='right',fontsize=8,color='darkred')
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(path,dpi=170);plt.close(fig)


def replay_figures(replay_dir,source='synthetic'):
    p=Path(replay_dir);rows=read_jsonl(p/'curves.jsonl')
    for axis,filename,label in [('cumulative_accelerator_seconds','search-cost.png','Replayed allocated accelerator-seconds'),('trial','search-trials.png','Revealed trials')]:
        fig,ax=plt.subplots(figsize=(7,4))
        palette={'default':'#555555','random':'#2878b5','gp':'#8e5aa9','neural':'#168a76'}
        linestyles={'default':'-','random':'-','gp':'--','neural':'-.'}
        for method in ('default','random','gp','neural'):
            lines=[r for r in rows if r['method']==method]
            for i,seed in enumerate(sorted({r['seed'] for r in lines})):
                pts=[r for r in lines if r['seed']==seed]
                ax.step([r[axis] for r in pts],[r['best_feasible_goodput'] for r in pts],where='post',alpha=.7,color=palette[method],linestyle=linestyles[method],marker='o' if method=='default' else None,markersize=4,label=method if i==0 else None)
        ax.set(xlabel=label,ylabel='Best feasible observed quality goodput (answers/s)',title='Offline-table search replay');ax.legend();ax.grid(alpha=.2)
        _save(fig,p/filename,source)


def prediction_figure(directory):
    p=Path(directory);rows=read_jsonl(p/'heldout-predictions.jsonl')
    fig,axs=plt.subplots(1,2,figsize=(9,4))
    for ax,target,pred,label in [(axs[0],'actual_goodput','goodput_mean','Goodput (answers/s)'),(axs[1],'actual_p99','p99_mean','P99 E2E (s)')]:
        valid=[r for r in rows if r[target] is not None]
        if valid:
            a=[r[target] for r in valid];b=[r[pred] for r in valid];ax.scatter(a,b,s=20,alpha=.6)
            lo,hi=min(a+b),max(a+b);ax.plot([lo,hi],[lo,hi],'k--',lw=1)
        ax.set(xlabel='Observed '+label,ylabel='Predicted '+label)
    fig.suptitle('Held-out workload diagnostic');_save(fig,p/'prediction-error.png',rows[0]['source'])


def measured_figures(directory):
    p=Path(directory);rows=read_jsonl(p/'observations.jsonl') if (p/'observations.jsonl').exists() else []
    if any(r['source']!='measured' for r in rows): raise ValueError('measured plot refuses synthetic data')
    fields=['run_id','status','scenario_id','config_id','offered','completed','accuracy','quality_goodput','p99_e2e_s','p99_completed_e2e_s','peak_memory_bytes','accelerator_seconds']
    with (p/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in rows:
            merged={**r,**r['metrics'],**r['cost']};writer.writerow({k:merged.get(k) for k in fields})
    valid=[r for r in rows if r['status']=='ok']
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for r in valid:
        rate=r['workload']['arrival_rate'];m=r['metrics'];label=r['run_id']
        axes[0].scatter(rate,m['completed_requests_per_s'],label=label)
        if m.get('p99_e2e_s') is not None:axes[1].scatter(rate,m['p99_e2e_s'],label=label)
    if not valid:
        for ax in axes:ax.text(.5,.5,'No completed serving trial',ha='center',transform=ax.transAxes)
    axes[0].set(xlabel='Offered target rate (requests/s)',ylabel='Completed requests/s');axes[1].set(xlabel='Offered target rate (requests/s)',ylabel='All-offered P99 E2E (s)')
    fig.suptitle('Development pilot — insufficient for tail-latency claims');_save(fig,p/'load-latency.png','measured')
    fig,ax=plt.subplots(figsize=(7,4))
    for r in valid:ax.scatter(r['cost']['accelerator_seconds'],r['metrics']['accuracy'],label=r['run_id'])
    ax.set(xlabel='Total trial allocated accelerator-seconds (includes startup/drain)',ylabel='Full-cohort development accuracy',ylim=(0,1),title='Development accuracy and actual trial cost')
    if valid:ax.legend(fontsize=6)
    _save(fig,p/'accuracy-cost.png','measured')
    for r in valid:
        d=p/'runs'/r['run_id'];telemetry=read_jsonl(d/'telemetry.jsonl');queue=read_jsonl(d/'queue.jsonl')
        fig,axes=plt.subplots(2,1,figsize=(8,5),sharex=True)
        if queue:axes[0].step([x['time_s'] for x in queue],[x['client_outstanding'] for x in queue],where='post',label='Client outstanding')
        waiting=[s for s in telemetry if 'vllm:num_requests_waiting' in s]
        if waiting:axes[0].plot([s['time_s'] for s in waiting],[s['vllm:num_requests_waiting'] for s in waiting],label='Engine waiting')
        mem=[s for s in telemetry if s.get('device_memory_bytes') is not None]
        if mem:axes[1].plot([s['time_s'] for s in mem],[s['device_memory_bytes']/2**30 for s in mem])
        else:axes[1].text(.5,.5,'Device memory time series unavailable on this run',ha='center',transform=axes[1].transAxes)
        axes[0].set(ylabel='Requests');axes[0].legend();axes[1].set(xlabel='Seconds from load start',ylabel='Allocated device GiB');fig.suptitle(r['run_id'])
        _save(fig,d/'stability.png','measured')
