#include "local_proto.h"

/* for qsort, order list by row and col */
int by_row_col(const void *ii, const void *jj)
{
    const struct order *i = ii, *j = jj;

    if (i->row != j->row)
        return i->row - j->row;

    return (i->col - j->col);
}

/* for qsort, order list by cat */
int by_cat(const void *ii, const void *jj)
{
    const struct order *i = ii, *j = jj;

    if (i->cat < j->cat)
        return -1;

    return (i->cat > j->cat);
}

/* for bsearch, find cat */
int srch_cat(const void *pa, const void *pb)
{
    int *p1 = (int *)pa;
    int *p2 = (int *)pb;

    if (*p1 < *p2)
        return -1;
    if (*p1 > *p2)
        return 1;
    return 0;
}
