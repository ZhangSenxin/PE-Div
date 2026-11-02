"""Final ASOP panel and information logos; no historical caches required."""
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Arc, Circle, PathPatch
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Affine2D
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict

AA = 'ACDEFGHIKLMNPQRSTVWY'
MAX_BITS = math.log2(20)
COLORS = {'exp1_only': '#46A9E0', 'exp2_only': '#EF5C55', 'overlap': '#A3DACA'}
AA_COLORS = {a: ('#267343' if a in 'GSTC' else '#4256A5' if a in 'HKR' else
                 '#C44141' if a in 'DE' else '#8D4D9B' if a in 'NQ' else '#333333') for a in AA}
GLYPHS = {a: TextPath((0, 0), a, size=1, prop=FontProperties(family='DejaVu Sans', weight='bold')) for a in AA}
GLYPHS['I'] = TextPath((0, 0), 'I', size=1, prop=FontProperties(family='DejaVu Sans Mono', weight='bold'))

def clean(ax):
    # Preserve the already approved paper panel's axes and typography.
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(width=.7, length=3)

def color(x):
    return to_hex(CMAP(NORM(x)))

def save(fig, output, stem, dpi):
    fig.savefig(output / (stem + '.png'), dpi=dpi, facecolor='white')
    fig.savefig(output / (stem + '.pdf'), facecolor='white')
    plt.close(fig)

def spread(ax,rows):
    """Disperse identical positions on a one-marker-radius circle; allow overlap."""
    ax.figure.canvas.draw();groups={};result=[]
    radius=MARKER_RADIUS*ax.figure.dpi/72
    for r in rows:groups.setdefault((r['signed_ASOP'],r['exclusive_nodes']),[]).append(r)
    for (true_x,true_y),group in sorted(groups.items()):
        group=sorted(group,key=lambda r:r['cluster_id']);m=len(group)
        if m==1:angles=[0]
        elif true_x<=-1+1e-12:angles=np.linspace(-np.pi/2,np.pi/2,m)
        elif true_x>=1-1e-12:angles=np.linspace(np.pi/2,3*np.pi/2,m)
        else:angles=np.arange(m)*2*np.pi/m
        raw=np.array([true_x,true_y]);p=ax.transData.transform(raw)
        for r,angle in zip(group,angles):
            delta=radius*np.array([np.cos(angle),np.sin(angle)]) if m>1 else np.zeros(2)
            chosen=p+delta;x,y=ax.transData.inverted().transform(chosen)
            result.append(dict(cluster_id=r['cluster_id'],true_x=float(true_x),true_y=int(true_y),display_x=float(x),display_y=float(y),
                               offset_x_points=float(delta[0]*72/ax.figure.dpi),offset_y_points=float(delta[1]*72/ax.figure.dpi),expanded=m>1))
    return result

def colorbar(fig,ax,compact=False):
    p=ax.get_position()
    cax=fig.add_axes([p.x0+p.width*.08,.105,p.width*.84,.022])
    cb=fig.colorbar(ScalarMappable(norm=NORM,cmap=CMAP),cax=cax,orientation='horizontal',ticks=[-1,-.5,0,.5,1])
    cb.ax.tick_params(labelsize=7,length=2,width=.5);cb.outline.set_linewidth(.5)
    if not compact:cb.set_label('Color: signed ASOP',fontsize=8,labelpad=3)
    return cax

def panel(ax,rows,selected,annotated=False,compact=False):
    ax.set(xlim=SETTINGS['x_limits'],ylim=SETTINGS['y_limits'],yscale='log',xticks=[-1,-.5,0,.5,1],
           xlabel='Signed antigen-selection origin purity (ASOP)',ylabel='Exclusive sequences per family (A + B)',title='Family size and antigen-selection origin purity')
    ax.set_yticks(SETTINGS['y_ticks']);ax.yaxis.set_major_formatter(ScalarFormatter());clean(ax)
    for x in SETTINGS['reference_x']:ax.axvline(x,color='#939BA2',ls='--',lw=.8,zorder=1)
    layout=spread(ax,rows);lookup={r['cluster_id']:r for r in rows}
    for p in layout:
        if p['expanded']:
            ax.plot([p['true_x'],p['display_x']],[p['true_y'],p['display_y']],color='#A5ADB4',lw=.45,zorder=2)
            ax.scatter([p['true_x']],[p['true_y']],s=7,marker='+',c='#777777',linewidths=.5,zorder=2)
    for name in ['red','green','blue']:
        pp=[p for p in layout if lookup[p['cluster_id']]['region']==name]
        ax.scatter([p['display_x'] for p in pp],[p['display_y'] for p in pp],s=MARKER_AREA,
                   c=[color(p['true_x']) for p in pp],alpha=.85,edgecolors='none',linewidths=0,zorder=3)
    if annotated:
        by_id={p['cluster_id']:p for p in layout}
        for r in selected:
            p=by_id[r['cluster_id']];target=r['target_x']
            ax.annotate(r['cluster_id'],xy=(p['display_x'],p['display_y']),xytext=(8,14) if target==-1 else (0,14),textcoords='offset points',
                        fontsize=7,color='#263542',ha='left' if target==-1 else 'center',va='bottom',arrowprops=dict(arrowstyle='-',lw=.7,color='#555555'),
                        bbox=dict(boxstyle='round,pad=.15',facecolor='white',edgecolor='none',alpha=.9),zorder=5)
    if not compact:ax.text(0,1.10,f'{len(rows)} families; coincident points spread by one marker radius',transform=ax.transAxes,fontsize=7.5,color='#555555')
    return layout

def logo_matrix(seqs):
    arr=np.array([list(s) for s in seqs]);n,L=arr.shape
    counts=np.array([[(arr[:,p]==a).sum() for a in AA] for p in range(L)])
    frequencies=counts/n
    logp=np.zeros_like(frequencies);np.log2(frequencies,out=logp,where=frequencies>0)
    entropy=-(frequencies*logp).sum(axis=1)
    info=np.maximum(0,MAX_BITS-entropy)
    heights=frequencies*info[:,None]
    assert np.all(counts.sum(axis=1)==n)
    assert np.allclose(heights.sum(axis=1),info)
    assert np.all((info>=0)&(info<=MAX_BITS+1e-12))
    return dict(aa_order=AA,n_sequences=n,counts=counts.tolist(),frequencies=frequencies.tolist(),entropy_bits=entropy.tolist(),information_bits=info.tolist(),letter_heights_bits=heights.tolist())

def draw_logo(ax,logo):
    matrix=np.array(logo['letter_heights_bits']);L=len(matrix)
    for pos,heights in enumerate(matrix):
        bottom=0
        for j in np.argsort(heights,kind='stable'):
            height=heights[j]
            if height<=1e-10:continue
            a=AA[j];glyph=GLYPHS[a];bb=glyph.get_extents()
            trans=Affine2D().translate(-bb.x0,-bb.y0).scale(.88/bb.width,height/bb.height).translate(pos+1-.44,bottom)
            ax.add_patch(PathPatch(glyph,transform=trans+ax.transData,facecolor=AA_COLORS[a],edgecolor='none'))
            bottom+=height
    ax.set(xlim=(.4,L+.6),ylim=(0,4.55),xticks=range(1,L+1),yticks=[0,1,2,3,4],xlabel='CDR3 amino-acid position',ylabel='Information (bits)')
    ax.set_title(f'Information logo | n = {logo["n_sequences"]} unique sequences | uniform 20-AA background',fontsize=10,pad=9)
    ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=9)

def draw_tree_logo(row,coords,edges,logo,pdf,png):
    fig=plt.figure(figsize=(10,12));ax=fig.add_axes([.055,.29,.89,.635]);ax.set_aspect('equal');ax.axis('off')
    node={r['sequence_id']:r for r in coords};children=defaultdict(list)
    for u,v,w in edges:
        if node[u]['radius_aa']>node[v]['radius_aa']:u,v=v,u
        assert node[v]['radius_aa']-node[u]['radius_aa']==w
        children[u].append(v)
    for parent,kids in children.items():
        p=node[parent];r=p['radius_aa'];theta=p['theta_radians']
        if len(kids)==1:
            c=node[kids[0]];ax.plot([p['x'],c['x']],[p['y'],c['y']],c='#777777',lw=.55)
        else:
            arc_r=r+.5;angles=[node[k]['theta_radians'] for k in kids]
            ax.plot([p['x'],arc_r*np.cos(theta)],[p['y'],arc_r*np.sin(theta)],c='#777777',lw=.55)
            ax.add_patch(Arc((0,0),2*arc_r,2*arc_r,theta1=np.degrees(min(angles)),theta2=np.degrees(max(angles)),color='#777777',lw=.55))
            for k in kids:
                c=node[k];t=c['theta_radians'];ax.plot([arc_r*np.cos(t),c['x']],[arc_r*np.sin(t),c['y']],c='#777777',lw=.55)
    for c in coords:ax.add_patch(Circle((c['x'],c['y']),.15,color=COLORS[c['category']],zorder=3))
    extent=max(max(abs(c['x']),abs(c['y'])) for c in coords)+1
    ax.set_xlim(-extent,extent);ax.set_ylim(-extent,extent)
    fig.suptitle(f'1 aa | {row["cluster_id"]} | n = {row["total_nodes"]:,}',fontsize=16,y=.973)
    fig.text(.5,.944,f'Antigen-selection origin purity (ASOP) = {row["ASOP"]:.3f}',ha='center',fontsize=11)
    handles=[Line2D([],[],marker='o',ls='',color=COLORS[c],label=f'{c}: {row[c]}') for c in COLORS]
    fig.legend(handles=handles,loc='center',bbox_to_anchor=(.5,.267),ncol=3,frameon=False,fontsize=10)
    logoax=fig.add_axes([.085,.072,.86,.145]);draw_logo(logoax,logo)
    fig.text(.5,.018,'Equal weight per unique sequence; uniform 20-AA background; no small-sample correction',ha='center',fontsize=8,color='#666666')
    pdf.savefig(fig)
    fig.savefig(png,dpi=300,facecolor='white');plt.close(fig)


def plot_asop(data, settings, output):
    """Return displayed positions separately from the scientific coordinates."""
    global SETTINGS, MARKER_AREA, MARKER_RADIUS, CMAP, NORM
    SETTINGS = settings
    MARKER_AREA = settings['marker_area']
    MARKER_RADIUS = math.sqrt(MARKER_AREA) / 2
    CMAP = LinearSegmentedColormap.from_list('DC_red_green_blue', settings['palette'], N=settings['color_levels'])
    NORM = Normalize(vmin=-1, vmax=1)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
                         'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'axes.linewidth': .7, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    rows = [dict(region='red' if f['signed_ASOP'] < -.5 else 'blue' if f['signed_ASOP'] > .5 else 'green', **f)
            for f in data['families'] if f['plotted']]
    fig, ax = plt.subplots(figsize=settings['figsize'])
    fig.subplots_adjust(left=.17, right=.96, bottom=.30, top=.84)
    layout = panel(ax, rows, data['selected_examples'], annotated=True)
    colorbar(fig, ax)
    save(fig, output, 'family_size_signed_asop_annotated', settings['png_dpi'])
    return layout


def tree_coordinates(data, family):
    """Original one-hot-centroid root and descendant-leaf circular layout."""
    indices = family['indices']
    nodes = [data['union'][i]['sequence_id'] for i in indices]
    index_set = set(indices)
    edges = [(data['union'][u]['sequence_id'], data['union'][v]['sequence_id'], w)
             for u, v, w in data['global_mst_edges'] if u in index_set and v in index_set and w <= 1]
    if len(edges) != len(nodes) - 1:
        raise AssertionError('Family tree must have n-1 edges')
    sequences = [data['union'][i]['seq'] for i in indices]
    array = np.frombuffer(''.join(sequences).encode(), dtype=np.uint8).reshape(len(indices), -1)
    score = np.zeros(len(indices))
    for column in range(array.shape[1]):
        count = np.bincount(array[:, column], minlength=256)
        score += count[array[:, column]] / len(indices)
    root = nodes[int(np.argmax(score))]
    adjacency = {node: [] for node in nodes}
    for u, v, weight in edges:
        adjacency[u].append((v, weight))
        adjacency[v].append((u, weight))
    distance, children, front = {root: 0}, {}, [root]
    order = []
    while front:
        next_front = []
        for u in front:
            order.append(u)
            children[u] = []
            for v, weight in sorted(adjacency[u]):
                if v not in distance:
                    distance[v] = distance[u] + weight
                    children[u].append(v)
                    next_front.append(v)
        front = next_front
    # Iterative DFS preserves the original lexicographic leaf order.
    stack, leaves, ranges = [(root, False)], [], {}
    while stack:
        node, postorder = stack.pop()
        kids = sorted(children[node])
        if postorder:
            ranges[node] = (ranges[kids[0]][0], ranges[kids[-1]][1])
        elif not kids:
            ranges[node] = (len(leaves), len(leaves))
            leaves.append(node)
        else:
            stack.append((node, True))
            stack.extend((child, False) for child in reversed(kids))
    coords = []
    for index, node in zip(indices, nodes):
        lo, hi = ranges[node]
        theta = (lo + hi) / 2 * 2 * np.pi / len(leaves)
        radius = distance[node]
        coords.append(dict(sequence_id=node, radius_aa=float(radius), theta_radians=float(theta),
                           x=float(radius * np.cos(theta)), y=float(radius * np.sin(theta)),
                           category=data['union'][index]['category'], is_root=node == root))
    return coords, edges, logo_matrix(sequences)


def plot_examples(data, output):
    target = output / 'selected_families'
    target.mkdir()
    coordinates, logo_sites = [], []
    for family in data['selected_examples']:
        coords, edges, logo = tree_coordinates(data, family)
        coordinates.extend(dict(cluster_id=family['cluster_id'], **c) for c in coords)
        for position, (entropy, information, freq) in enumerate(zip(
                logo['entropy_bits'], logo['information_bits'], logo['frequencies']), 1):
            logo_sites.extend(dict(cluster_id=family['cluster_id'], position=position, amino_acid=aa,
                                   frequency=float(p), entropy_bits=entropy, information_bits=information,
                                   letter_height_bits=float(p * information)) for aa, p in zip(AA, freq))
        with PdfPages(target / (family['cluster_id'] + '_tree_with_logo.pdf')) as pdf:
            draw_tree_logo(family, coords, edges, logo, pdf, target / (family['cluster_id'] + '_tree_with_logo.png'))
    return coordinates, logo_sites
