"""Measured-data-only figures; no fabricated successful curves or hidden filtering."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .report import frame

_PROFILE_NOTE=''


def save(fig,path):
    if _PROFILE_NOTE:
        fig.suptitle(_PROFILE_NOTE,fontsize=10,color='firebrick')
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'),dpi=180)
    fig.savefig(path.with_suffix('.svg'))
    plt.close(fig)


def make_plots(cfg,run):
    global _PROFILE_NOTE
    _PROFILE_NOTE='Implementation / exploratory check — not a formal research result' if cfg['profile']!='formal' else ''
    out = Path(run)/'plots'
    out.mkdir(parents=True,exist_ok=True)
    errors=[]
    for seed in cfg['seeds']:
        root=Path(run)/f'seed_{seed}'
        directory=root/'evaluation'
        if not (directory/'complete.json').exists():
            continue
        fx=frame(directory/'effects.csv')
        fx['arm']=fx.arm.astype(str).str.zfill(2)
        clean=fx[(fx.source=='cache_summary')&(fx.cohort=='ordinary')&~fx.reset]
        density=clean[(clean.family_model=='density2')&clean.arm.isin(['10','01'])]
        fig,ax=plt.subplots(figsize=(6,5))
        for arm,g in density.groupby('arm'):
            sampled=g.sample(min(len(g),5000),random_state=17)
            ax.scatter(sampled.actual_effect,sampled.predicted_effect,s=6,alpha=.25,label=arm)
        ax.plot([-1,1],[-1,1],'k--',lw=1)
        ax.set(xlabel='Actual change in answer-1 probability',ylabel='Predicted change',title=f'Seed {seed}: held-out partial cache effects')
        ax.legend(title='Earlier/current swap')
        save(fig,out/f'seed_{seed}_effect_predictions')
        bars=clean[(clean.continuation=='?')&(clean.recipient_clue==0)&(clean.family_model=='density2')]
        fig,ax=plt.subplots(figsize=(7,4))
        arms=['00','10','01','11'];x=np.arange(4)
        actual=[bars[bars.arm==a].actual_p1.mean() for a in arms]
        predicted=[bars[bars.arm==a].predicted_p1.mean() for a in arms]
        ax.bar(x-.18,actual,.36,label='Transformer');ax.bar(x+.18,predicted,.36,label='Density surrogate')
        ax.set(xticks=x,xticklabels=['Original','Earlier swap','Current swap','Full donor'],ylabel='Probability of answer 1',title=f'Seed {seed}: same supplied query')
        ax.legend();save(fig,out/f'seed_{seed}_memory_swaps')
        joints=frame(directory/'joints.csv')
        for family in cfg['analysis']['families']:
            sub=joints[(joints.source=='cache_summary')&(joints.family_model==family)&(joints.cohort=='ordinary')&~joints.reset&(joints.horizon==3)]
            effect=clean[(clean.family_model==family)&clean.arm.isin(['10','01'])]
            errors.append(dict(seed=seed,family_model=family,joint3_tv=sub.joint_tv.mean(),effect_error=effect.effect_error.mean()))
        geometry=frame(directory/'geometry.csv')
        g=geometry[geometry.source=='cache_summary']
        fig,ax=plt.subplots(figsize=(6,5))
        for reset,gg in g.groupby('reset'):
            ax.scatter(gg.trace_distance,gg.future3_pair_tv,s=9,alpha=.3,label=f'Reset={reset}')
        ax.set(xlabel='Distance between encoded density states',ylabel='Transformer three-token future TV',title=f'Seed {seed}: geometry versus measured behavior')
        ax.legend();save(fig,out/f'seed_{seed}_geometry')
        timeline=frame(directory/'timeline.csv')
        if len(timeline):
            first_family=timeline.family.iloc[0]
            selected=timeline[timeline.family==first_family]
            fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
            for ax,reset in zip(axes,[False,True]):
                for clue,color in [(0,'royalblue'),(1,'orangered')]:
                    gg=selected[(selected.reset==reset)&(selected.clue==clue)].sort_values('offset')
                    ax.plot(gg.offset,gg.actual_p1,'o-',color=color,label=f'Transformer, clue {clue}')
                    ax.plot(gg.offset,gg.predicted_p1,'--',color=color,label=f'Surrogate, clue {clue}')
                ax.set(xlabel='Tokens supplied after initial encoding',ylabel='Probability of next token 1',
                       title='Reset control' if reset else 'Paired histories',ylim=(-.03,1.03))
                ax.set_xticks(range(8),['root','a','b','a','b','a','b','?'])
            axes[0].legend(fontsize=8)
            save(fig,out/f'seed_{seed}_prediction_timeline')
        calibration=frame(directory/'calibration.csv')
        if len(calibration):
            fig,ax=plt.subplots(figsize=(6,5))
            for family,gg in calibration[calibration.source=='cache_summary'].groupby('family_model'):
                ax.plot(gg.mean_predicted,gg.mean_actual,'o-',label=family)
            ax.plot([0,1],[0,1],'k--',lw=1)
            ax.set(xlabel='Binned predicted answer-1 probability',ylabel='Mean transformer answer-1 probability',
                   title=f'Seed {seed}: intervention prediction calibration')
            ax.legend();save(fig,out/f'seed_{seed}_calibration')
        fig,ax=plt.subplots(figsize=(6,4))
        for family,gg in joints[(joints.source=='cache_summary')&~joints.reset&(joints.horizon==3)].groupby('family_model'):
            means=gg.groupby('delay').joint_tv.mean()
            ax.plot(means.index,means.values,'o-',label=family)
        ax.axvspan(13,33,color='grey',alpha=.08,label='Unfitted long-delay region')
        ax.set(xlabel='Filler delay before comparison',ylabel='Mean joint-three-token TV error',title=f'Seed {seed}: delay generalization')
        ax.legend(fontsize=8);save(fig,out/f'seed_{seed}_delay_generalization')
        cost=frame(directory/'benchmark.csv')
        if len(cost):
            fig,axes=plt.subplots(1,2,figsize=(11,4))
            for ax,batch in zip(axes,[1,256]):
                selected=cost[(cost.source=='cache_summary')&(cost.batch_size==batch)&cost.operation.isin(['encode','update','predict'])]
                table=selected.pivot(index='family_model',columns='operation',values='seconds_per_call')*1000
                table.plot.bar(ax=ax)
                ax.set(ylabel='Milliseconds per batch',title=f'Batch {batch}');ax.tick_params(axis='x',rotation=20)
            save(fig,out/f'seed_{seed}_latency')
        interaction=frame(directory/'interactions.csv')
        if len(interaction):
            selected=interaction[(interaction.source=='cache_summary')&(interaction.family_model=='density2')&(interaction.cohort=='ordinary')&~interaction.reset]
            fig,ax=plt.subplots(figsize=(6,4))
            selected.assign(magnitude=selected.interaction.abs()).groupby('continuation').magnitude.mean().plot.bar(ax=ax)
            ax.set(xlabel='Common continuation',ylabel='Mean absolute two-factor contrast',title=f'Seed {seed}: non-additivity of cache effects')
            ax.tick_params(axis='x',rotation=0);save(fig,out/f'seed_{seed}_interactions')
        updates=frame(directory/'updates.csv')
        if len(updates):
            fig,ax=plt.subplots(figsize=(6,4))
            for family in cfg['analysis']['families']:
                u=updates[(updates.source=='cache_summary')&(updates.family_model==family)&(updates.cohort=='ordinary')&~updates.reset]
                grouped=u.groupby('steps').output_tv.mean()
                ax.plot(grouped.index,grouped.values,'o-',label=family)
            ax.set(xlabel='Supplied tokens since initial encoding',ylabel='Mean full-distribution TV',title=f'Seed {seed}: updates without teacher refresh')
            ax.legend();save(fig,out/f'seed_{seed}_autonomous_updates')
        logs=root/'teacher'/'train.jsonl'
        if logs.exists():
            rows=[json.loads(line) for line in logs.read_text().splitlines()]
            fig,ax=plt.subplots(figsize=(6,4))
            ax.plot([r['step'] for r in rows],[r['loss'] for r in rows],lw=.7)
            ax.set(xlabel='Optimizer step',ylabel='Next-token cross-entropy',title=f'Seed {seed}: transformer training')
            save(fig,out/f'seed_{seed}_training')
        data=np.load(directory/'density_trajectories.npz')
        if data['coordinates'].size:
            import plotly.graph_objects as go
            coords,labels=data['coordinates'],data['labels']
            fig=go.Figure()
            u,v=np.mgrid[0:2*np.pi:32j,0:np.pi:20j]
            fig.add_trace(go.Surface(x=np.cos(u)*np.sin(v),y=np.sin(u)*np.sin(v),z=np.cos(v),opacity=.08,showscale=False,colorscale='Greys'))
            for clue,color in [(0,'royalblue'),(1,'orangered')]:
                sel=labels[:,1]==clue
                fig.add_trace(go.Scatter3d(x=coords[sel,0],y=coords[sel,1],z=coords[sel,2],mode='markers',name=f'Clue {clue}',
                                          marker=dict(size=3,color=color,opacity=.6),
                                          text=[f'family={r[0]}, reset={r[2]}, offset={r[3]}, stage={r[4]}, position={r[5]}' for r in labels[sel]]))
            fig.update_layout(title=f'Seed {seed}: fitted density coordinates (compare within matched stage/position)',
                              scene=dict(xaxis_title='r_x',yaxis_title='r_y',zaxis_title='r_z',aspectmode='cube'))
            fig.write_html(out/f'seed_{seed}_bloch_ball.html',include_plotlyjs=True)
        diagnostic=frame(root/'diagnostics'/'held_out_posterior.csv')
        if len(diagnostic):
            grouped=diagnostic.groupby(['source','mode']).absolute_error.mean().unstack()
            fig,ax=plt.subplots(figsize=(7,4));grouped.plot.bar(ax=ax)
            ax.set(ylabel='Mean posterior absolute error',title=f'Seed {seed}: native versus compressed information')
            save(fig,out/f'seed_{seed}_posterior_diagnostic')
    if errors:
        df=pd.DataFrame(errors)
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,metric,title in zip(axes,['joint3_tv','effect_error'],['Three-token future error','Intervention-effect error']):
            grouped=df.groupby('family_model')[metric]
            names=list(grouped.mean().index)
            ax.bar(names,grouped.mean().values,yerr=grouped.std().fillna(0).values,capsize=4)
            for name,g in grouped:
                ax.scatter([names.index(name)]*len(g),g.values,color='black',s=15)
            ax.set(title=title,ylabel='Error (lower is better)');ax.tick_params(axis='x',rotation=20)
        save(fig,out/'baseline_comparison')
    (out/'README.md').write_text('Figures use measured run outputs only. PNG and SVG figures are paired; Bloch plots are self-contained HTML.\n'
                                 'Matrix coordinates are not unique semantic axes. Across-seed gauges are not aligned.\n'
                                 'Baseline error bars show variation across teacher seeds, not a three-seed population significance claim.\n')
