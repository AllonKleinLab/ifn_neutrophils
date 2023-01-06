
# Dependencies
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import datetime
from itertools import product
import re
from time import time
from typing import Optional, Union

#List of definitions

##  spring_cmap
##  save_spring
##  sctransform
##  clr_transform
##  knn_label_smoothing
##  print_etime
##  now
##  expression_centroids
##  multinomial_naive_bayes
##  chunk_list
##  chunk_array
##  fix_filename
##  cell_classifier
##  parallel_cell_classifier
##  umap_plot
##  save_df_as_npz
##  bivariate_scatter
##  label_correspondence


spring_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('spring_cmap', 
                                             [(0,    '#000000'),
                                              (1,    '#00ff00')], N=256)


def now():
    """
    current date and time as filename-friendly string
    """
    return datetime.datetime.now().strftime('%y%m%d_%Hh%M')


def print_etime(start):
    """
    Print elapsed time given an initial timestamp.
    """
    etime = time() - start
    print('Elapsed time: {m} minutes and {s} seconds.'.format(m=int(etime//60), s=round(etime%60,1)))

def fix_filename(string):
    return re.sub(r'\W', '', str(string).replace(' == ','_').replace(' & ','_').replace(' ', '_'))

def save_spring(anndata, project_dir, sample_name, embedding_name='untitled', embedding_use='X_umap', use_raw=False, web=False, timestamp=False, remove_barcodes=True, **kwargs):
    """
    anndata (AnnData) -> AnnData object to export
    project_dir (str) -> directory where Spring files will be saved in a subdirectory */spring
    sample_name (str) -> name of the subfolder within spring
    embedding_name (str) -> name of specific embedding of the sample (def. "untitled")
    embedding_use (str) -> key from spring_data.obsm to use to construct coordinates
    timestamp (bool) -> appends timestamp to directory name
    remove_barcodes (bool) -> Remove .obs columns containing the string 'barcode'
    

    """
    
    from scipy.sparse.csr import csr_matrix
    from os import makedirs, rename
    from json import loads, dump
    from time import time
    

    start = time()
    
    spring_data = anndata.copy()
    
    if use_raw == False:
        try:
            del spring_data.raw
        except:
            pass
        
    if remove_barcodes:
        spring_data.obs = spring_data.obs.loc[:,~spring_data.obs.columns.str.contains('barcode')]
    
    if project_dir[-1] != '/':
        project_dir = project_dir+'/'
    
    if timestamp:
        subp = embedding_name+'_{}'.format(now())
    else:
        subp = embedding_name
        
    spring_dir = project_dir+sample_name

    makedirs(spring_dir, exist_ok=True)
    spring_data.X = csr_matrix(spring_data.X)



    sc.external.exporting.spring_project(spring_data, # adata object
                                spring_dir, # main SPRING directory for this dataset
                                embedding_use, # 2-D embedding to use for output
                                subplot_name = subp, **kwargs) # SPRING subplot directory for this particular embedding/parameter set

    
    
    # correct colors for all keys
    keys_with_colors = [k.split('_colors')[0] for k in anndata.uns.keys() if 'colors' in k]
    
    if len(keys_with_colors) > 0:
        with open('{}/categorical_coloring_data.json'.format(spring_dir+'/'+subp), 'r') as jfile:
                    jdata = jfile.read()
                    jdict = loads(jdata)

        for i, key in enumerate(keys_with_colors):
            try:
                updated_colors = {label_name:anndata.uns[key+'_colors'][i].lower() for i,(label_name,old_color) in enumerate(jdict[key]['label_colors'].items())}
                jdict[key]['label_colors'] = updated_colors
            except:
                pass

        with open('{}/categorical_coloring_data.json'.format(spring_dir+'/'+subp), 'w') as jfile:
            dump(jdict, jfile)

    
    
    if web:
        with open('{}/categorical_coloring_data.json'.format(spring_dir+'/'+subp), 'r') as jfile:
            jdata = jfile.read()
            jdict = loads(jdata)

        jdf = pd.DataFrame([np.array((k, *v['label_list'])) for k,v in jdict.items() if k != 'barcode']).set_index(0)
        jdf[~jdf.index.str.contains('barcode')].to_csv('{}/categorical_data.csv'.format(spring_dir+'/'+subp), header=False)

    coordinates_df = pd.read_csv('{}/coordinates.txt'.format(spring_dir+'/'+subp), header=None)
    coordinates_df = pd.read_csv('{}/coordinates.txt'.format(spring_dir+'/'+subp), header=None)
    coordinates_df.iloc[:,2] = -coordinates_df.iloc[:,2]
    
    rename('{}/coordinates.txt'.format(spring_dir+'/'+subp), '{}/coordinates_flipped.txt'.format(spring_dir+'/'+subp))
    coordinates_df.to_csv('{}/coordinates.txt'.format(spring_dir+'/'+subp), header=False, float_format='%.6f', index=False)

    print_etime(start)
    
    
    # to do: add actual parameters from spring_data.uns

    if web:
        print('''Expression data (Required): counts_norm.npz

        Gene list (Required): genes.txt

        Cell groupings (Optional): categorical_data.csv

        Color tracks (Optional): color_data_gene_sets.csv

        Custom coordinates (Optional): coordinates_mirrored.txt

        Minimum cell counts 0

        Minimum cells expressing >= 3 counts per gene 3

        Variability percentile 50

        PCA components 160

        Neighbors 10''')
        
        
def clr_transform(X):
    '''
    implements the CLR transform used in CITEseq (need to confirm in Seurat's code)
    https://doi.org/10.1038/nmeth.4380
    '''
    import scanpy as sc
    from scipy.sparse.csr import csr_matrix
    
    if isinstance(X, sc.AnnData):
        X = X.to_df()
    else:
        X = X.copy()
        
    if isinstance(X, csr_matrix):
        X = X.todense()
    
    X_log1p = np.log1p(X)
    X_clr = X_log1p - X_log1p.mean(axis=1)[:, None]

    return X_clr


def sctransform(adata, hvg = False, n_genes = 4000, rlib_loc = '', copy=True, min_cells = 5, batch_var = None, verbose = True, key_added = 'sct', correct_umi=True, v2_flavor = False):
    """
    Function to call scTransform normalization or HVG selection from Python. Modified from https://github.com/normjam/benchmark/blob/master/normbench/methods/ad2seurat.py. 
    
    Parameters
    ----------
    adata: `AnnData`
        AnnData object of RNA counts.
    hvg: `boolean`
        Should the hvg method be used (returning a reduced adata object) or the normalization method (returning a normalized adata). 
    n_genes: `int`
        Number of hvgs to return if the hvg method is selected. A selection of 4000-5000 generally yields the best results. 
    rlib_loc: `str`
        R library location that will be added to the default .libPaths() to locate the required packages. 
  
    Returns
    -------
    returns an AnnData object reduced to the highly-variable genes. 
    """
    import importlib
    
    rpy2_import = importlib.util.find_spec('rpy2')
    if rpy2_import is None:
        raise ImportError(
            "deviance requires rpy2. Install with pip install rpy2")
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    import anndata2ri
    from scipy.sparse import issparse
    
    ro.globalenv['rlib_loc'] = rlib_loc
    ro.r('.libPaths(c(rlib_loc, .libPaths()))')
    ro.r('suppressPackageStartupMessages(library(Seurat))')
    ro.r('suppressPackageStartupMessages(library(scater))')
    anndata2ri.activate()
    
    correct_umi_bool = 'TRUE' if correct_umi else 'FALSE'
    vst_flavor = "vst.flavor = 'v2'," if v2_flavor else ''
    
    if copy:
        adata = adata.copy()        
    
    
    sc.pp.filter_genes(adata, min_cells = min_cells)
    
    if issparse(adata.X):
        if not adata.X.has_sorted_indices:
            adata.X.sort_indices()

    for key in adata.layers:
        if issparse(adata.layers[key]):
            if not adata.layers[key].has_sorted_indices:
                adata.layers[key].sort_indices()
    
    ro.globalenv['adata'] = adata
    ro.globalenv['min_cells'] = min_cells

    ro.r('seurat_obj = as.Seurat(adata, counts="X", data = NULL)')
    ro.r('seurat_obj <- SeuratObject::RenameAssays(seurat_obj, originalexp = "RNA")')
    
    if verbose:
        ro.globalenv['verbose'] = True
    else:
        ro.globalenv['verbose'] = False
        
    
    
    if hvg:
        numpy2ri.activate()
        ro.globalenv['n_genes'] = n_genes
        
        print('Reducing the data to', n_genes, 'variable genes.')
        
        
        if batch_var == None: #bypass Seurat check
            ro.r(f'res <- SCTransform(object=seurat_obj, return.only.var.genes = TRUE, min_cells = min_cells, {vst_flavor} do.correct.umi = {correct_umi_bool}, verbose = verbose, variable.features.n = n_genes)')
        else:
            ro.globalenv['batch_var'] = batch_var
            ro.r(f'res <- SCTransform(object=seurat_obj, return.only.var.genes = TRUE, min_cells = min_cells, {vst_flavor} do.correct.umi = {correct_umi_bool}, verbose = verbose, batch_var = batch_var, variable.features.n = n_genes)')
            
        hvgs_r =  ro.r('res@assays$SCT@var.features')
        adata = adata[:,list(hvgs_r)]
        adata.var['highly_variable'] = True
        return adata
    else: 
        
        if batch_var == None: #bypass Seurat check
            ro.r(f'res <- SCTransform(object=seurat_obj, return.only.var.genes = FALSE, {vst_flavor} do.correct.umi = {correct_umi_bool}, verbose = verbose, min_cells = min_cells)')
        else:
            ro.globalenv['batch_var'] = batch_var
            ro.r(f'res <- SCTransform(object=seurat_obj, return.only.var.genes = FALSE, {vst_flavor} do.correct.umi = {correct_umi_bool}, verbose = verbose, min_cells = min_cells,  batch_var = batch_var)')
        
        
        sct_x = ro.r('res@assays$SCT@scale.data').T
        norm_x = ro.r('res@assays$SCT@counts').T
        log1p_x = ro.r('res@assays$SCT@data').T

        adata.layers[key_added] = sct_x
        
        if correct_umi:
            adata.layers[key_added+'_umi'] = norm_x
            adata.layers[key_added+'_log1p'] = log1p_x
        
        ro.pandas2ri.activate()
        adata.var = adata.var.join(ro.r('res@assays$SCT@SCTModel.list$model1@feature.attributes'))
        ro.pandas2ri.deactivate()
        
        adata.var['highly_variable'] = adata.var_names.isin(adata.var.sort_values('residual_variance', ascending=False).index[:n_genes])
        anndata2ri.activate()
#         adata.raw = adata
        

        return adata
    

def knn_label_smoothing(adata, label_col, weighted=True, k=None, likelihood_matrix=None, return_df = False):
    """
    Returns a Pandas Series of dtype="category" where each observation's label has been voted by its nearest neighbors based on adata.obsp['connectivities'].
    
    adata (sc.AnnData) -> AnnData object with neighbors keys.
    
    label_col (str) -> Name of the column to perform smoothing on.
    
    weighted (bool) -> Default: True. Whether to weigh neighbor voting by connectivity or consider all neighbors equal.
    
    k (int; optional) -> If specified, truncates connectivities up to each cell's kth largest value.
    
    likelihood_matrix (pd.DataFrame; optional) -> If specified, performs smoothing on the full likelihood matrix rather than each cell's most likely class. Must agree in index with adata.
    
    """

    import bottleneck as bn
    from sklearn.preprocessing import LabelBinarizer
    from scipy.sparse.csr import csr_matrix
    
    
    #check sparsity
    if isinstance(adata.obsp['connectivities'], csr_matrix):
        all_connectivities = adata.obsp['connectivities'].todense().copy()
    else:
        all_connectivities = adata.obsp['connectivities'].copy()

    # OPTIONAL: truncate connectivities up to kth largest value
    if k != None:
        kth_largest_connectivity = bn.partition(all_connectivities, all_connectivities.shape[0] - k)[:,-k:].min(1)
        connectivities_larger_than_kth = all_connectivities >= kth_largest_connectivity
        truncated_knn = np.multiply(all_connectivities, connectivities_larger_than_kth)
    else:
        truncated_knn = all_connectivities
        
    if not weighted:
        truncated_knn[truncated_knn > 0] = 1

    # OPTIONAL: provide likelihood matrix to weigh in
    # TODO: fix df * np.ndarray operability
    if likelihood_matrix:
        knn_weighted_labels = truncated_knn @ likelihood_matrix
    else:
        # binarize labels to enable weighted knn voting
        lb = LabelBinarizer()
        binarized_labels = lb.fit_transform(adata.obs[label_col])
        knn_weighted_labels = truncated_knn @ binarized_labels
    
    # smoothed labels as one-hot-encoded matrix
    binarized_knn_labels = (knn_weighted_labels == knn_weighted_labels.max(axis=1)).astype(int)
    
    if return_df:
        return pd.DataFrame(knn_weighted_labels, index=adata.obs_names, columns= lb.classes_)
    
    # invert one-hot-encoding into labels; save as categorical Series
    # TODO: debug case where two labels have equal score
    smoothed_labels = pd.Series(pd.Categorical(lb.inverse_transform(binarized_knn_labels).reshape(-1), 
                         categories = adata.obs[label_col].cat.categories), 
                         index=adata.obs_names)
    
    return smoothed_labels

def expression_centroids(adata, group_by):
    cell_groups = [lab for lab, df in adata.obs.groupby(group_by)]

    if len(group_by) == 1:
        group_by = group_by[0]

    if type(group_by) == str:
        return pd.DataFrame(np.vstack([adata[adata.obs[group_by] == lab].X.mean(0) for i,lab in enumerate(cell_groups)]), index=cell_groups, columns=adata.var_names)

    # if group with multiple categories
    else:
        return pd.DataFrame(np.vstack([adata[adata.obs.groupby(group_by).apply(lambda x: x.index.tolist())[i]].X.mean(0) for i,lab in enumerate(cell_groups)]), index=cell_groups, columns=adata.var_names)
    
def multinomial_naive_bayes(op,cp): #ported from rz_functions.bayesian_classifier
    '''
    op - observed gene expression profile, genes x samples
    cp - class profiles, genes x samples, same genes as op
    returns log10(P(E|type)), the max value is the closes cell type
    '''
    
    #we assume that each cell type has a well define centroid, let's represent this expression vector
    #as the fractions of all mRNAs for each genes (i.e. normalized the expression such that the expression of
    #all genes sums to 1)
    
    cp = cp/cp.sum()
    
    #we assume that the exact expression pattern we observe (E) is multinomially distributed around the centroid.
    #Bayes' formula: P(type|E) = P(E|type)*P(type)/P(E)
    #our classifier is naive, so each E is equally likely (this is how I interpret "naive", although it
    #may have more to do with the assumption that genes are uncorrelated)
    
    ptes = pd.DataFrame({cell:(np.log10(cp.T.values)*op[cell].values).sum(axis=1) for cell in op.columns})
    ptes.index = cp.columns
    return ptes

def save_df_as_npz(obj, filename): #ported from rz_functions.save_df
    np.savez_compressed(filename, data=obj.values, index=obj.index.values, columns=obj.columns.values)
    

# based on: https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
def chunk_array(arr, n, axis=0):
    """Yield successive n-sized chunks from arr."""
    if axis == 0:
        for i in range(0, arr.shape[0], n):
            yield arr[i:i + n]
    elif axis == 1:
        for j in range(0, arr.shape[1], n):
            yield arr[:, j:j + n]
    else:
        raise Exception('Invalid axis')
        
def umap_plot(anndata_raw, color = None, vmax = None, vmin=1e-5, folder='figures', filt='', split_by_cats = '', legend_loc = 'on data', legend_fontsize = 6, legend_fontoutline= 2, dpi_show = 150, dpi_save = 500, norm=None, cmap='viridis', show_fig=True, save_fig=False, filename_prefix='', filename_suffix='', return_fig=True, basis='X_umap', use_raw = False, custom_series = None, highlight_categorical = None, rc_context_kwargs={'figure.figsize' : (4,4)}, **kwargs):
    """
    Wrapper for scanpy.pl.umap with additional functionalities:
        * split plot by categories and visualize percentiles of the joint dataset
        * highlight the localization of each value in a categorical
        * apply filters to the data through Pandas-styled queries
        * plot custom Pandas Series without the need to add them to the AnnData object
        
    In addition, it sets certain aesthetic choices by default:
        * square plot boxes
        * legends on the data, with white outlines
        * DPI of 150 for plotting, 500 for saving (PDF)
        
    
    Parameters
    ----------
    anndata_raw : AnnData
        Annotated data matrix.
    color : typing.Union[str, typing.Sequence[str], NoneType], optional (default: None)
    Keys for annotations of observations/cells or variables/genes, e.g.,
    `'ann1'` or `['ann1', 'ann2']`.
    vmax : typing.Union[float, str]
    
    vmin : typing.Union[float, str]
    
    ... work in progress
    """    
    
    if type(color) == str:
        color = [color]
    
    if filt:
        anndata = anndata_raw[anndata_raw.obs.eval(filt)].copy()
    else:
        anndata = anndata_raw.copy()
    
    if custom_series.__str__() != 'None': # the Python equivalent of hearing nails scraping on a blackboard:
        
        if type(custom_series) == dict:
            
            if color == None:
                color = []
            
            for k, series in custom_series.items():
                anndata.obs[k] = series
        
                color.append(k)
                    
    if highlight_categorical:
        
        if color == None:
            color = []
        
        if type(highlight_categorical) == str:
            is_category = {}           
            for cat in anndata.obs[highlight_categorical].astype('category').cat.categories:
                anndata.obs[f'is_{cat}'] = (anndata.obs[highlight_categorical] == cat).map({True : cat, False : ''})
                color.append(f'is_{cat}')
            
        if type(highlight_categorical) == dict:
            for col, groups_to_highlight in highlight_categorical.items():
                for cat in groups_to_highlight:
                    anndata.obs[f'is_{cat}'] = (anndata.obs[col] == cat).map({True : cat, False : ''})
                    color.append(f'is_{cat}')

    if split_by_cats: # if prompted to split plots by categories, map a list of n categories to 2**n queries and plots
        if type(split_by_cats) == str:

            cat_a = split_by_cats
            queries = ['{} == "{}"'.format(cat_a, x) for x in anndata.obs[cat_a].cat.categories]

        if type(split_by_cats) == list:
            if len(split_by_cats) == 1:

                cat_a = split_by_cats[0]
                queries = ['{} == "{}"'.format(cat_a, x) for x in anndata.obs[cat_a].cat.categories]

            elif len(split_by_cats) == 2:

                cat_a, cat_b = split_by_cats
                queries = {'{} == "{}" & {} == "{}"'.format(cat_a, x, cat_b, y) for x,y in  list(product(anndata.obs[cat_a].cat.categories, anndata.obs[cat_b].cat.categories))}

            else:
                raise Exception('Too many arguments.')
    else: queries = False
    
    qdict = {}
    
    if queries:
        for q in queries:
            qdict[q] = anndata[anndata.obs.eval(q)]

    if len(qdict) == 0:
        qdict[fix_filename(color)] = anndata
        
    qindex = 1
    fig_list = []
    
    
    for qtitle, qdata in qdict.items():
        
        with plt.rc_context(rc_context_kwargs):
            
            # if embedding is split by categories, calculate percentiles from the shared percentile across groups
            if split_by_cats and (type(vmax) == str):
                try:
                    shared_percentile = float(vmax.split('p')[1])
                except:
                    raise Exception('Percentile needs to be formatted as "pXX". For example: p80.')
                
                
                if type(color) == list:
                    vmax_param = [np.percentile(anndata.X[:,anndata.var_names == gene_name].todense(), shared_percentile) for gene_name in color]
                    
                    vmax_param = []
                    
                    for gene_name in color:
                        if gene_name in anndata.var_names:
                            vmax_param.append(np.percentile(anndata.X[:,anndata.var_names == gene_name].todense(), shared_percentile))
                        else:
                            vmax_param.append(np.percentile(anndata.obs[gene_name], shared_percentile))
                    
                # shouldn't ever be needed
                elif type(color) == str:
                    if color in anndata.var_names:
                        vmax_param = np.percentile(anndata.X[:,anndata.var_names == color].todense(), shared_percentile)
                    else:
                        vmax_param = np.percentile(anndata.obs[color], shared_percentile)
            else:
                vmax_param = vmax
                
            if norm != None:
                vmax_param = None
                vmin = None
                        
            qfig = sc.pl.embedding(qdata, 
                       cmap = cmap,
                       basis = basis,
                       norm = norm,
                       color = color,
                       vmin = vmin,
                       vmax = vmax_param,                               
                       legend_loc = legend_loc, 
                       legend_fontsize = legend_fontsize, 
                       legend_fontoutline = legend_fontoutline,
                       use_raw = use_raw,
                       return_fig=return_fig, **kwargs)


            if return_fig:
                plt.suptitle(qtitle, y=1.05, fontweight = 'bold')
                qfig.set_dpi(dpi_show)

        if save_fig:
            if qtitle != fix_filename(color):
                figtitle = filename_prefix + '_'*(len(filename_prefix) != 0) + fix_filename(filt) + '_'*(len(filt)!=0) + fix_filename(qtitle) + '_by_' + fix_filename(color) + '_'*(len(filename_suffix) != 0) + filename_suffix
            else:
                figtitle = filename_prefix + '_'*(len(filename_prefix) != 0) + fix_filename(filt) + '_'*(len(filt)!=0) + fix_filename(color) + '_'*(len(filename_suffix) != 0) + filename_suffix

            figname = '{}/{}_{}_{}.pdf'.format(folder, figtitle, now(), qindex)

            qfig.savefig(fname=figname, dpi = dpi_save, bbox_inches='tight')
            print('{} done!'.format(figname))
        
        
        if show_fig == False:
            plt.close()
        
        qindex += 1
        fig_list.append(qfig)
        
    return fig_list

def cell_classifier(adata, bdata, state_column_b, comment, save_dir, pseudo_count = 0.1, step = 5000, verbose = True, return_most_likely=True, normalization_constant=1e4, progress_bar = False):
    """
    adata (AnnData) - dataset to create labels for
    bdata (AnnData or DataFrame) - dataset from which to compute centroids (or dataFrame of centroids)
    state_column_b (str, iterable) - column(s) from obs to deduce cell groupings, compute centroids, and define likelihoods
    comment (str) - string to be appended to the name of the array storing log-likelihoods
    save_dir (str) - directory to save results
    pseudo_count (float) - value to be added to centroids to avoid division by zero
    step (int) - number of cells to classify in a given batch
    verbose (bool) - Default: True
    return_most_likely (bool) - Return Pandas Series based on the index of adata with the argmax class for each cell barcode. Otherwise, return DataFrame with log-likelihoods. Default: True.
    """
    
    # pre-processing of A, computation of centroids of dataset B
    sc.pp.filter_genes(adata, min_cells=1)  # prevent division by zero
    sc.pp.normalize_total(adata, target_sum=normalization_constant)

    adata_X = adata.X
    
    if type(bdata) == pd.DataFrame:
        b_centroids = bdata
    else:
        sc.pp.filter_genes(bdata, min_cells=1) # prevent division by zero
        sc.pp.normalize_total(bdata, target_sum=normalization_constant)
        b_centroids = expression_centroids(bdata, state_column_b).T + pseudo_count
        
    gene_list = adata.var_names
    
    
    # determine genes to be considered for classification
    
    # common genes
    common_genes = np.in1d(gene_list, b_centroids.index) # what genes in b are in a? (boolean array)

    # genes detected in the current dataset:
    detected_genes = np.array(adata_X.sum(axis=0))[0]>0 # what genes in a have more than 0 counts across all cells? (boolean array)

    # combine masks
    gene_mask = common_genes & detected_genes
    common_genes = gene_list[gene_mask]

    if verbose:
        print('No. of genes in data set to be classified: {}.'.format(len(gene_list)))
        print('No. of genes in data set to use as template for classification: {}.'.format(b_centroids.shape[0]))
        print('No. of genes present/detected in both data sets: {}.'.format(len(common_genes)))
        
    
    # classification loop with progress bar
    if progress_bar:
        from tqdm import tqdm
        with tqdm(total=adata_X.shape[0], unit='cells') as progress_bar:

            i = 0
            step = step
            logl_arrays = []

            for j in range(step,adata_X.shape[0]+step,step):

                j = min(j,adata_X.shape[0])

                # slice cells from the expression matrix in groups of stepsize (from i to j, which are updated in the loop)
                # consider only common genes as defined above

                a_slice_ij = pd.DataFrame(adata_X.T[gene_mask,i:j].todense(), index=common_genes)
                logl_array_ij = multinomial_naive_bayes(a_slice_ij,b_centroids.loc[common_genes])
                logl_arrays.append(logl_array_ij)


                i0 = i
                i = j

                progress_bar.update(j-i0)
    else:
        i = 0
        step = step
        logl_arrays = []

        for j in range(step,adata_X.shape[0]+step,step):

            j = min(j,adata_X.shape[0])

            # slice cells from the expression matrix in groups of stepsize (from i to j, which are updated in the loop)
            # consider only common genes as defined above

            a_slice_ij = pd.DataFrame(adata_X.T[gene_mask,i:j].todense(), index=common_genes)
            logl_array_ij = multinomial_naive_bayes(a_slice_ij,b_centroids.loc[common_genes])
            logl_arrays.append(logl_array_ij)


            i0 = i
            i = j
        
        
    # save and return results
    
    # concatenate
    logl_df = pd.concat(logl_arrays,axis=1)

    # reset index
    logl_df.columns = np.arange(logl_df.shape[1])

    if save_dir[-1] == '/':
        save_dir = save_dir[:-1]
    
    filename = save_dir+'/loglikelihoods_multinomial_naive_bayes_%s_%s'%(comment,now())
    print('Class log-likelihoods stored at '+filename+'.npz')
    save_df_as_npz(logl_df,filename)
    
    
    if return_most_likely:
        return logl_df.idxmax().values
    else:
        return logl_df
    
    
def parallel_cell_classifier(adata, bdata, comment, save_dir,  state_column_b='', pseudo_count = 0.1, step = 5000, verbose = True, return_most_likely=True, nproc=4, progress_bar=True, normalization_constant = 1e4):
    """
    adata (AnnData) - dataset to create labels for
    bdata (AnnData or DataFrame) - dataset from which to compute centroids (or dataFrame of centroids)
    state_column_b (str, iterable) - column(s) from obs to deduce cell groupings, compute centroids, and define likelihoods
    comment (str) - string to be appended to the name of the array storing log-likelihoods
    save_dir (str) - directory to save results
    pseudo_count (float) - value to be added to centroids to avoid division by zero
    step (int) - number of cells to classify in a given batch
    verbose (bool) - Default: True
    return_most_likely (bool) - Return Pandas Series based on the index of adata with the argmax class for each cell barcode. Otherwise, return DataFrame with log-likelihoods. Default: True.
    nproc (int) - Number of processes to run
    """
    from multiprocessing import Pool
    
    # pre-processing of A, computation of centroids of dataset B
    sc.pp.filter_genes(adata, min_cells=1)  # prevent division by zero
    sc.pp.normalize_total(adata, target_sum=normalization_constant)

    adata_X = adata.X
    
    if type(bdata) == pd.DataFrame:
        b_centroids = bdata
    else:
        sc.pp.filter_genes(bdata, min_cells=1) # prevent division by zero
        sc.pp.normalize_total(bdata, target_sum=normalization_constant)
        b_centroids = expression_centroids(bdata, state_column_b).T + pseudo_count
        
    gene_list = adata.var_names
    
    
    # determine genes to be considered for classification
    
    # common genes
    common_genes = np.in1d(gene_list, b_centroids.index) # what genes in b are in a? (boolean array)

    # genes detected in the current dataset:
    detected_genes = np.array(adata_X.sum(axis=0))[0]>0 # what genes in a have more than 0 counts across all cells? (boolean array)

    # combine masks
    gene_mask = common_genes & detected_genes
    common_genes = gene_list[gene_mask]

    if verbose:
        print('No. of genes in data set to be classified: {}.'.format(len(gene_list)))
        print('No. of genes in data set to use as template for classification: {}.'.format(b_centroids.shape[0]))
        print('No. of genes present/detected in both data sets: {}.'.format(len(common_genes)))
          
        
    #define slices    
    adata_X_slices = [pd.DataFrame(slice_ij, index=common_genes) for slice_ij in chunk_array(adata_X[:,gene_mask].todense().T, step, axis=1)]
    
    #define function to parallelize
    global classify_on_b # multiprocessing/pickle requires top level function
    def classify_on_b(slice_ij): 
        return multinomial_naive_bayes(slice_ij,b_centroids.loc[common_genes])

    
    #run 
    with Pool(processes=nproc) as pool:              # start 4 worker processes
    #logl_arrays = pool.imap_unordered(classify_on_b, adata_X_slices)

        logl_arrays = []
        
                    
        if progress_bar:
            from tqdm import tqdm
            with tqdm(total = adata_X.shape[0], unit='cells') as pbar:
                for i, logl in tqdm(enumerate(pool.imap(classify_on_b, adata_X_slices))):
                    pbar.update(adata_X_slices[i].shape[1])
                    logl_arrays.append(logl)
        else:
            for i, logl in enumerate(pool.imap(classify_on_b, adata_X_slices)):
                logl_arrays.append(logl)
            
        
    # save and return results
    
    # concatenate
    logl_df = pd.concat(logl_arrays,axis=1)

    # reset index
    logl_df.columns = np.arange(logl_df.shape[1])

    if save_dir[-1] == '/':
        save_dir = save_dir[:-1]
    
    filename = save_dir+'/loglikelihoods_multinomial_naive_bayes_%s_%s'%(comment,now())
    print('Class log-likelihoods stored at '+filename+'.npz')
    save_df_as_npz(logl_df,filename)
    
    
    if return_most_likely:
        return logl_df.idxmax().values
    else:
        return logl_df
    
def bivariate_scatter(anndata, var1, var2, color1 = 'blue', color2 = 'red', color_addition = None, 
                      cmap1 = None, cmap2 = None, cmap_steps = 256, basis = 'umap', var1_pmax=99,
                      s=10, edgecolors='k', lw=0.05, var2_pmax=99, use_raw=False, layer=None, var1_max=None, var2_max=None,
                      colorbar=True, return_fig=False, **kwargs):
    """
    Plot a bivariate scatterplot over an embedding of an AnnData object.
    
    Parameters:
    
    anndata (AnnData) -> AnnData object to extract information from
    var1 (str) -> Obs column or var name
    var2 (str) -> Obs column or var name
    color1 (str, list) -> color to generate colormap for var1
    color2 (str, list) -> color to generate colormap for var2
    color_addition (str) -> Color addition method. Default is "light" for color gradients and "average" for color maps. Accepted methods: "light", "average", "product", "minimum"
    cmap1 (str) -> Name of the Matplotlib colormap to use for var1
    cmap2 (str) -> Name of the Matplotlib colormap to use for var2
    cmap_steps (int) -> Number of discrete bins to use for both colormaps
    basis (str) -> obsm key name after X_ to use as 2D embedding
    var1_pmax (int, float) -> Percentile to clip var1 values (default 99)
    var2_pmax (int, float)  -> Percentile to clip var2 values (default 99)
    use_raw (bool) -> Default: False
    layer (str) -> AnnData layer to extract expression values (overrides use_raw)
    colorbar (bool) -> Show colorbar?
    **kwargs -> Feed into plt.scatterplot
    
    """

    from sklearn.preprocessing import minmax_scale
    from scanpy.pl._tools.scatterplots import _get_color_source_vector


    feature1_raw = _get_color_source_vector(anndata, var1, use_raw = use_raw, layer = layer)
    feature2_raw = _get_color_source_vector(anndata, var2, use_raw = use_raw, layer = layer)
    
    feature1 = minmax_scale(feature1_raw)
    feature2 = minmax_scale(feature2_raw)
        
    feature1_min = feature1_raw.min()
    feature1_max = feature1_raw.max()
    feature2_min = feature2_raw.min()
    feature2_max = feature2_raw.max()
    
    
    if isinstance(var1_max,float) or isinstance(var1_max,int):    
        feature1[feature1_raw > var1_max] = 1
        feature1_max = var1_max

    elif isinstance(var1_pmax,float) or isinstance(var1_pmax,int):
        feature1[feature1 > np.percentile(feature1, var1_pmax)] = 1
        feature1_max = np.percentile(feature1_raw, var1_pmax)
   

    if isinstance(var2_max,float) or isinstance(var2_max,int): 
        feature2[feature2_raw > var2_max] = 1
        feature2_max = var2_max
    
    elif isinstance(var2_pmax,float) or isinstance(var2_pmax,int):
        feature2[feature2 > np.percentile(feature2, var2_pmax)] = 1
        feature2_max = np.percentile(feature2_raw, var2_pmax)
        
    def mix_colors(c1, c2, addition_method):
        
        if addition_method == 'light':
            c_out = np.sum([c1, c2], axis=0)
            c_out[c_out > 1] = 1
            return c_out
        
        if addition_method == 'product':
            c_out = c1*c2
            return c_out
        if addition_method == 'average':
            c_out = (c1+c2)/2
            return c_out
        if addition_method == 'minimum':
            c_out = np.minimum(c1, c2)
            return c_out
        if addition_method == 'sqrt':
            a = 2
            c_out = (0.5*c1**a + 0.5*c2**a)**(1/a)
            return c_out
    

    if cmap1 and cmap2:
        cmaps_provided = True

        if isinstance(cmap1, str):
            cmap1 = plt.colormaps.get(cmap1, cmap_steps)
        if isinstance(cmap2, str):
            cmap2 = plt.colormaps.get(cmap2, cmap_steps)

        c_array1 = cmap1(feature1)
        c_array2 = cmap2(feature2)
        
        if not color_addition:
            c_array = mix_colors(c_array1, c_array2, 'average')
        else:
            c_array = mix_colors(c_array1, c_array2, color_addition)            
    
    else:
        cmaps_provided = False
        
        if color_addition and (color_addition != 'light'):
            color0 = 'white'
        else:
            color0 = 'black'

        cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list(color1,[color0, color1], N=cmap_steps)
        cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list(color2,[color0, color2], N=cmap_steps)

        c_array1 = cmap1(feature1)
        c_array2 = cmap2(feature2)
                
        if not color_addition:
            c_array = mix_colors(c_array1, c_array2, 'light')
        else:
            c_array = mix_colors(c_array1, c_array2, color_addition)
        
        
    if colorbar:
        xx, yy = np.meshgrid(np.linspace(feature1_min,feature1_max, cmap_steps),np.linspace(feature2_min,feature2_max, cmap_steps))
        
        cbar_array1 = cmap1(minmax_scale(xx, axis=1))
        cbar_array2 = cmap2(minmax_scale(yy, axis=0))

        # Color for each point
        if cmaps_provided:
            if not color_addition:
                cbar_array = mix_colors(cbar_array1, cbar_array2, 'average')
            else:
                cbar_array = mix_colors(cbar_array1, cbar_array2, color_addition)
        else:
            if not color_addition:
                cbar_array = mix_colors(cbar_array1, cbar_array2, 'light')
            else:
                cbar_array = mix_colors(cbar_array1, cbar_array2, color_addition)



    if colorbar:
        fig, axes = plt.subplots(1,2, figsize=(6,4), dpi=200, gridspec_kw={'width_ratios': [4, 2]})
        axes[0].scatter(*anndata.obsm["X_{}".format(basis)].T, s=s, edgecolors=edgecolors, lw=lw, c= c_array, **kwargs)
        
        axes[0].set_xlabel("X_{}1".format(basis))
        axes[0].set_ylabel("X_{}2".format(basis))

        axes[1].imshow(cbar_array, origin='lower')
        
        axes[1].set_xticks(np.linspace(0, cmap_steps, 4))
        axes[1].set_yticks(np.linspace(0, cmap_steps, 4))
        
        cbar_xticklabels = (axes[1].get_xticks() / cmap_steps  * (feature1_max - feature1_min) + feature1_min).round(1)
        cbar_yticklabels = (axes[1].get_yticks() / cmap_steps  * (feature2_max - feature2_min) + feature2_min).round(1)
        
        axes[1].set_xticklabels(cbar_xticklabels)
        axes[1].set_yticklabels(cbar_yticklabels)
            
        axes[1].set_xlabel(var1)
        axes[1].set_ylabel(var2)
        
        plt.tight_layout()
        
        axes[0].axis('square')

    else:
        fig, ax = plt.subplots(1, figsize=(4,4), dpi=200)
        ax.scatter(*anndata.obsm["X_{}".format(basis)].T, s=s, edgecolors=edgecolors, lw=lw, c= c_array, **kwargs)
        ax.axis('square')
        ax.set_xlabel("X_{}1".format(basis))
        ax.set_ylabel("X_{}2".format(basis))
        
    
    if return_fig: return fig
    
def bivariate_scatter(anndata, var1, var2, color1 = 'blue', color2 = 'red', color_addition = None, 
                      cmap1 = None, cmap2 = None, cmap_steps = 256, basis = 'umap', var1_pmax=99,
                      s=10, edgecolors='k', lw=0.05, var2_pmax=99, use_raw=False, layer=None, var1_max=None, var2_max=None,
                      colorbar=True, return_fig=False, **kwargs):
    """
    Plot a bivariate scatterplot over an embedding of an AnnData object.
    
    Parameters:
    
    anndata (AnnData) -> AnnData object to extract information from
    var1 (str) -> Obs column or var name
    var2 (str) -> Obs column or var name
    color1 (str, list) -> color to generate colormap for var1
    color2 (str, list) -> color to generate colormap for var2
    color_addition (str) -> Color addition method. Default is "light" for color gradients and "average" for color maps. Accepted methods: "light", "average", "product", "minimum"
    cmap1 (str) -> Name of the Matplotlib colormap to use for var1
    cmap2 (str) -> Name of the Matplotlib colormap to use for var2
    cmap_steps (int) -> Number of discrete bins to use for both colormaps
    basis (str) -> obsm key name after X_ to use as 2D embedding
    var1_pmax (int, float) -> Percentile to clip var1 values (default 99)
    var2_pmax (int, float)  -> Percentile to clip var2 values (default 99)
    use_raw (bool) -> Default: False
    layer (str) -> AnnData layer to extract expression values (overrides use_raw)
    colorbar (bool) -> Show colorbar?
    **kwargs -> Feed into plt.scatterplot
    
    """

    from sklearn.preprocessing import minmax_scale
    from scanpy.pl._tools.scatterplots import _get_color_source_vector


    feature1_raw = _get_color_source_vector(anndata, var1, use_raw = use_raw, layer = layer)
    feature2_raw = _get_color_source_vector(anndata, var2, use_raw = use_raw, layer = layer)
    
    feature1 = minmax_scale(feature1_raw)
    feature2 = minmax_scale(feature2_raw)
        
    feature1_min = feature1_raw.min()
    feature1_max = feature1_raw.max()
    feature2_min = feature2_raw.min()
    feature2_max = feature2_raw.max()
    
    
    if isinstance(var1_max,float) or isinstance(var1_max,int):    
        feature1[feature1_raw > var1_max] = 1
        feature1_max = var1_max

    elif isinstance(var1_pmax,float) or isinstance(var1_pmax,int):
        feature1[feature1 > np.percentile(feature1, var1_pmax)] = 1
        feature1_max = np.percentile(feature1_raw, var1_pmax)
   

    if isinstance(var2_max,float) or isinstance(var2_max,int): 
        feature2[feature2_raw > var2_max] = 1
        feature2_max = var2_max
    
    elif isinstance(var2_pmax,float) or isinstance(var2_pmax,int):
        feature2[feature2 > np.percentile(feature2, var2_pmax)] = 1
        feature2_max = np.percentile(feature2_raw, var2_pmax)
        
    def mix_colors(c1, c2, addition_method):
        
        if addition_method == 'light':
            c_out = np.sum([c1, c2], axis=0)
            c_out[c_out > 1] = 1
            return c_out
        
        if addition_method == 'product':
            c_out = c1*c2
            return c_out
        if addition_method == 'average':
            c_out = (c1+c2)/2
            return c_out
        if addition_method == 'minimum':
            c_out = np.minimum(c1, c2)
            return c_out
        if addition_method == 'sqrt':
            a = 2
            c_out = (0.5*c1**a + 0.5*c2**a)**(1/a)
            return c_out
    

    if cmap1 and cmap2:
        cmaps_provided = True

        if isinstance(cmap1, str):
            cmap1 = plt.colormaps.get(cmap1, cmap_steps)
        if isinstance(cmap2, str):
            cmap2 = plt.colormaps.get(cmap2, cmap_steps)

        c_array1 = cmap1(feature1)
        c_array2 = cmap2(feature2)
        
        if not color_addition:
            c_array = mix_colors(c_array1, c_array2, 'average')
        else:
            c_array = mix_colors(c_array1, c_array2, color_addition)            
    
    else:
        cmaps_provided = False
        
        if color_addition and (color_addition != 'light'):
            color0 = 'white'
        else:
            color0 = 'black'

        cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list(color1,[color0, color1], N=cmap_steps)
        cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list(color2,[color0, color2], N=cmap_steps)

        c_array1 = cmap1(feature1)
        c_array2 = cmap2(feature2)
                
        if not color_addition:
            c_array = mix_colors(c_array1, c_array2, 'light')
        else:
            c_array = mix_colors(c_array1, c_array2, color_addition)
        
        
    if colorbar:
        xx, yy = np.meshgrid(np.linspace(feature1_min,feature1_max, cmap_steps),np.linspace(feature2_min,feature2_max, cmap_steps))
        
        cbar_array1 = cmap1(minmax_scale(xx, axis=1))
        cbar_array2 = cmap2(minmax_scale(yy, axis=0))

        # Color for each point
        if cmaps_provided:
            if not color_addition:
                cbar_array = mix_colors(cbar_array1, cbar_array2, 'average')
            else:
                cbar_array = mix_colors(cbar_array1, cbar_array2, color_addition)
        else:
            if not color_addition:
                cbar_array = mix_colors(cbar_array1, cbar_array2, 'light')
            else:
                cbar_array = mix_colors(cbar_array1, cbar_array2, color_addition)



    if colorbar:
        fig, axes = plt.subplots(1,2, figsize=(6,4), dpi=200, gridspec_kw={'width_ratios': [4, 2]})
        axes[0].scatter(*anndata.obsm["X_{}".format(basis)].T, s=s, edgecolors=edgecolors, lw=lw, c= c_array, **kwargs)
        
        axes[0].set_xlabel("X_{}1".format(basis))
        axes[0].set_ylabel("X_{}2".format(basis))

        axes[1].imshow(cbar_array, origin='lower')
        
        axes[1].set_xticks(np.linspace(0, cmap_steps, 4))
        axes[1].set_yticks(np.linspace(0, cmap_steps, 4))
        
        cbar_xticklabels = (axes[1].get_xticks() / cmap_steps  * (feature1_max - feature1_min) + feature1_min).round(1)
        cbar_yticklabels = (axes[1].get_yticks() / cmap_steps  * (feature2_max - feature2_min) + feature2_min).round(1)
        
        axes[1].set_xticklabels(cbar_xticklabels)
        axes[1].set_yticklabels(cbar_yticklabels)
            
        axes[1].set_xlabel(var1)
        axes[1].set_ylabel(var2)
        
        plt.tight_layout()
        
        axes[0].axis('square')

    else:
        fig, ax = plt.subplots(1, figsize=(4,4), dpi=200)
        ax.scatter(*anndata.obsm["X_{}".format(basis)].T, s=s, edgecolors=edgecolors, lw=lw, c= c_array, **kwargs)
        ax.axis('square')
        ax.set_xlabel("X_{}1".format(basis))
        ax.set_ylabel("X_{}2".format(basis))
        
    
    if return_fig: return fig
    
def label_correspondence(anndata, label_column, loglikelihoods):
    """
    anndata (Anndata)
    label_column (str)
    logls_npz (str)
    logls_df (Pandas DataFrame)
    """
    
    if type(anndata) == str:
        anndata = sc.read(anndata)
    
    if type(loglikelihoods) == str:
        logls_df = pd.DataFrame(**np.load(loglikelihoods, allow_pickle=True))
    elif type(loglikelihoods) == pd.DataFrame:
        logls_df = loglikelihoods
    else:
        raise Exception('Invalid format for log-likelihoods. Please provide a path to a npz file or a Pandas dataframe object.')

    likelihoods_rescaled = np.exp(logls_df - logls_df.max(0))
    likelihoods_normalized = likelihoods_rescaled / likelihoods_rescaled.sum(0)
    labels_and_likelihoods = pd.concat((anndata.obs[label_column].reset_index(drop=True), likelihoods_normalized.T), axis=1)
    
    return labels_and_likelihoods.groupby(label_column).agg('mean').div(labels_and_likelihoods.groupby(label_column).agg('mean').sum(1), axis=0)