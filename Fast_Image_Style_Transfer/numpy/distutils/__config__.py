# This file is generated by /builddir/build/BUILD/python27-numpy-1.7.2-8.16.amzn1/setup.py
# It contains system_info results at the time of building this package.
__all__ = ["get_info","show"]

atlas_threads_info={'libraries': ['lapack', 'ptf77blas', 'ptcblas', 'atlas'], 'library_dirs': ['/usr/lib64/atlas'], 'language': 'f77', 'define_macros': [('ATLAS_INFO', '"\\"3.8.4\\""')], 'include_dirs': ['/usr/include']}
blas_opt_info={'libraries': ['ptf77blas', 'ptcblas', 'atlas'], 'library_dirs': ['/usr/lib64/atlas'], 'language': 'c', 'define_macros': [('ATLAS_INFO', '"\\"3.8.4\\""')], 'include_dirs': ['/usr/include']}
atlas_blas_threads_info={'libraries': ['ptf77blas', 'ptcblas', 'atlas'], 'library_dirs': ['/usr/lib64/atlas'], 'language': 'c', 'define_macros': [('ATLAS_INFO', '"\\"3.8.4\\""')], 'include_dirs': ['/usr/include']}
lapack_opt_info={'libraries': ['lapack', 'ptf77blas', 'ptcblas', 'atlas'], 'library_dirs': ['/usr/lib64/atlas'], 'language': 'f77', 'define_macros': [('ATLAS_INFO', '"\\"3.8.4\\""')], 'include_dirs': ['/usr/include']}
lapack_mkl_info={}
blas_mkl_info={}
mkl_info={}

def get_info(name):
    g = globals()
    return g.get(name, g.get(name + "_info", {}))

def show():
    for name,info_dict in globals().items():
        if name[0] == "_" or type(info_dict) is not type({}): continue
        print(name + ":")
        if not info_dict:
            print("  NOT AVAILABLE")
        for k,v in info_dict.items():
            v = str(v)
            if k == "sources" and len(v) > 200:
                v = v[:60] + " ...\n... " + v[-60:]
            print("    %s = %s" % (k,v))
    