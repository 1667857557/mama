#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <stdint.h>

extern int32_t mamars_run(const double *, const double *, const double *,
                          size_t, size_t, double *, double *);
extern int32_t mamars_analyze(const double *, const double *, const double *,
    const double *, const double *, const double *, size_t, size_t,
    double *, double *, int32_t *, int32_t *);
extern int32_t mamars_fit(const double *,const double *,const double *,const double *,const double *,const double *,size_t,size_t,double *,double *,int32_t *,int32_t *,double *,double *,double *);

SEXP C_mamars_core(SEXP betas, SEXP omega, SEXP sigma) {
    SEXP dims = getAttrib(betas, R_DimSymbol);
    if (TYPEOF(betas) != REALSXP || XLENGTH(dims) != 2)
        error("betas must be a double matrix");
    size_t m = (size_t) INTEGER(dims)[0];
    size_t p = (size_t) INTEGER(dims)[1];
    if (XLENGTH(omega) != (R_xlen_t)(m * p * p) ||
        XLENGTH(sigma) != (R_xlen_t)(m * p * p))
        error("inconsistent native array dimensions");

    SEXP out_beta = PROTECT(allocMatrix(REALSXP, (int)m, (int)p));
    SEXP out_se = PROTECT(allocMatrix(REALSXP, (int)m, (int)p));
    int32_t rc = mamars_run(REAL(betas), REAL(omega), REAL(sigma), m, p,
                            REAL(out_beta), REAL(out_se));
    if (rc != 0) {
        UNPROTECT(2);
        error("MAMA calculation failed at SNP %d (singular or non-positive matrix)", rc);
    }
    SEXP result = PROTECT(allocVector(VECSXP, 2));
    SET_VECTOR_ELT(result, 0, out_beta);
    SET_VECTOR_ELT(result, 1, out_se);
    SEXP names = PROTECT(allocVector(STRSXP, 2));
    SET_STRING_ELT(names, 0, mkChar("beta"));
    SET_STRING_ELT(names, 1, mkChar("se"));
    setAttrib(result, R_NamesSymbol, names);
    UNPROTECT(4);
    return result;
}

SEXP C_mamars_fit(SEXP betas,SEXP ses,SEXP ld,SEXP lf,SEXP cf,SEXP sf){
 SEXP dims=getAttrib(betas,R_DimSymbol);size_t m=(size_t)INTEGER(dims)[0],p=(size_t)INTEGER(dims)[1];
 SEXP rb=PROTECT(allocMatrix(REALSXP,(int)m,(int)p)),rs=PROTECT(allocMatrix(REALSXP,(int)m,(int)p));
 SEXP keep=PROTECT(allocVector(LGLSXP,m)),tweak=PROTECT(allocVector(LGLSXP,m));
 SEXP lc=PROTECT(allocMatrix(REALSXP,p,p)),cc=PROTECT(allocMatrix(REALSXP,p,p)),sc=PROTECT(allocMatrix(REALSXP,p,p));
 int rc=mamars_fit(REAL(betas),REAL(ses),REAL(ld),REAL(lf),REAL(cf),REAL(sf),m,p,REAL(rb),REAL(rs),(int32_t*)LOGICAL(keep),(int32_t*)LOGICAL(tweak),REAL(lc),REAL(cc),REAL(sc));
 if(rc){UNPROTECT(7);error("native MAMA fit failed (code %d)",rc);}
 SEXP out=PROTECT(allocVector(VECSXP,7)),names=PROTECT(allocVector(STRSXP,7));SEXP values[]={rb,rs,keep,tweak,lc,cc,sc};const char*n[]={"beta","se","keep","tweaked","ld_coef","const_coef","se2_coef"};
 for(int i=0;i<7;i++){SET_VECTOR_ELT(out,i,values[i]);SET_STRING_ELT(names,i,mkChar(n[i]));}setAttrib(out,R_NamesSymbol,names);UNPROTECT(9);return out;
}

SEXP C_mamars_analysis(SEXP betas, SEXP ses, SEXP ld, SEXP lc, SEXP cc, SEXP sc) {
    SEXP dims = getAttrib(betas, R_DimSymbol);
    if (TYPEOF(betas) != REALSXP || XLENGTH(dims) != 2) error("invalid betas");
    size_t m = (size_t) INTEGER(dims)[0], p = (size_t) INTEGER(dims)[1];
    SEXP rb=PROTECT(allocMatrix(REALSXP,(int)m,(int)p));
    SEXP rs=PROTECT(allocMatrix(REALSXP,(int)m,(int)p));
    SEXP keep=PROTECT(allocVector(LGLSXP,(R_xlen_t)m));
    SEXP tweak=PROTECT(allocVector(LGLSXP,(R_xlen_t)m));
    int32_t rc=mamars_analyze(REAL(betas),REAL(ses),REAL(ld),REAL(lc),REAL(cc),REAL(sc),m,p,
                              REAL(rb),REAL(rs),(int32_t*)LOGICAL(keep),(int32_t*)LOGICAL(tweak));
    if(rc){UNPROTECT(4);error("native MAMA analysis failed (code %d)",rc);}
    SEXP result=PROTECT(allocVector(VECSXP,4));
    SET_VECTOR_ELT(result,0,rb);SET_VECTOR_ELT(result,1,rs);
    SET_VECTOR_ELT(result,2,keep);SET_VECTOR_ELT(result,3,tweak);
    SEXP names=PROTECT(allocVector(STRSXP,4));
    const char *n[]={"beta","se","keep","tweaked"};
    for(int i=0;i<4;i++)SET_STRING_ELT(names,i,mkChar(n[i]));
    setAttrib(result,R_NamesSymbol,names);UNPROTECT(6);return result;
}

static const R_CallMethodDef call_methods[] = {
    {"C_mamars_core", (DL_FUNC) &C_mamars_core, 3},
    {"C_mamars_analysis", (DL_FUNC) &C_mamars_analysis, 6},
    {"C_mamars_fit", (DL_FUNC) &C_mamars_fit, 6},
    {NULL, NULL, 0}
};

void R_init_mamars(DllInfo *dll) {
    R_registerRoutines(dll, NULL, call_methods, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
    R_forceSymbols(dll, TRUE);
}
