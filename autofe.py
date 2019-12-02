__doc__ = """
automated feature engineering. contains functions that, given continuous data,
will automatically generate nonlinear features that can be functions of just
each original feature themselves or be products of multiple features in order
to capture cross-feature effects. can be configured to generate different types
of features; by default, the function will just return the original DataFrame.

notes: only accepts pandas DataFrames. please make sure columns are continuous,
       or else you risk the output not really making any sense. when calling
       pair_nlfe, try to use as few options as possible in a call, as the
       underlying numpy library tries to allocate arrays using contiguous blocks
       of memory. both functions cause feature explosions, but pair_nlfe is
       an exceptional offender; you may get a memory allocation error if you
       have too many generated columns from pair_nlfe. also, the current
       implementation of nlfe and pair_nlfe is single-threaded; future versions
       may attempt to use the multiprocessing module for concurrent computation.

example:

suppose you have a DataFrame df with only continuous columns, and you wish to
compute the square and natural log of the feature columns and save the results
in a new DataFrame df_new. also, you would like to see what features were
created and how many features were created. thus, your function call would be

from autofe import nlfe
df_new = nlfe(df, poly2 = df.columns, ln = df.columns, verbose = True)

note that regardless of whether or not verbose == True, if there are any columns
that contain nonpositive values, nlfe will print a warning to stderr.

example:

going back to DataFrame df, suppose you want to compute the pairwise products of
the features and the natural log of 1 + the square of the pairwise products of
the features, saving the results in df_new. you do not want verbose output.
your corresponding function call would thus be

from autofe import pair_nlfe
df_new = pair_nlfe(df, poly1 = df.columns, lnquad = df.columns)
"""
# Changelog:
#
# 12-02-2019
#
# finished adding code for verbose option in pair_nlfe. also added examples for
# nlfe and pair_nlfe, and added some warnings on memory allocation errors.
#
# 12-01-2019
#
# happy december! now finals will soon be upon us...yikes. added _check_dfcols
# since i was just copying the same code to check the columns being passed for
# each named argument in nlfe. imported wrap from textwrap and defined the new
# KeyError_ class in order to get around the KeyError exception not parsing
# the newline correctly as i wanted a wrapped message to be passed. added the
# _pairprods function to compute products of column pairs, and finally finished
# pair_nlfe. tested nlfe and pair_nlfe on a modified version of X_train.csv to
# make sure that everything works as advertised. also added more functionality
# for the verbose option to nlfe; after every batch of created features, the
# names of the columns are printed (wrapped to 80 columns) to stdout and the
# total number of features computed is printed at the end. working on pair_nlfe.
#
# 11-30-2019
#
# completed code for nlfe. changed type requirement to strictly DataFrame only,
# and replaced some weird exception catching statements with if statements and
# manual exception raises as appropriate. changed docstring to reflect hard
# requirement for DataFrame.
#
# 11-29-2019
#
# continued writing docstrings for nlfe and pair_nlfe.
#
# 11-28-2019
#
# initial creation. happy turkey day everyone. changed name from
# feature_engineering.py, which i thought was too long, to autofe.py. really
# only wrote some initial docstrings for nlfe and pair_nlfe.

# define module name
_MODULE_NAME = "autofe"

# very strict on what can be passed
from pandas import Series, DataFrame, Index
from numpy import power, exp, sqrt
# we use the python math log since the numpy log does not throw ValueError
from math import log
from sys import stderr
# for line wrapping
from textwrap import wrap

# i made my own exception to get the error message to wrap correctly.
class KeyError_(KeyError):

    def __init__(self, msg): self.message = msg

    def __str__(self): return self.message

def _check_dfcols(data, cols, _fn, _colsn):
    """
    given a DataFrame data, list of columns cols, calling function name _fn, and
    name for cols _colsn, checks if any of the columns in cols are not in data.
    checks if cols is iterable. more importantly, will make a list of columns in
    cols not in data, and then raise a KeyError detailing the missing columns.
    """
    # save function name
    _fname = _check_dfcols.__name__
    # check if data is a DataFrame (in case the original check failed)
    if not isinstance(data, DataFrame):
        raise TypeError("{0}: data must be a pandas DataFrame".format(_fname))
    # check if cols, that has name _colsn, is an iterable but NOT a string
    if (not hasattr(cols, "__iter__")) or (isinstance(cols, str) == True):
        raise TypeError("{0}: {1} must be a list or list-like"
                        "".format(_fn, _colsn))
    # check if any columns in cols are not in data and add them to ucols
    ucols = []
    for col in cols:
        if col not in data.columns: ucols.append(col)
    # if ucols length > 0, then at least one column not in data. raise KeyError_
    # use initial_indent to offset the length of autofe.KeyError_, and then we
    # strip the whitespace in order to shrink the first line.
    if len(ucols) > 0:
        raise KeyError_("\n".join(wrap("{0}: {1}: unknown columns {2}"
                                       "".format(_fn, _colsn, ucols),
                                       width = 80, initial_indent = " " * 18)
        ).lstrip())
    # else just return None
    return None

def nlfe(data, verbose = False, poly2 = None, poly3 = None, mquad = None,
         ln = None, lnquad = None):
    """
    given a pandas Series or DataFrame of continuous data, create various
    nonlinear features from the data. can compute polynomial and power features,
    and logarithmic features (if values are positive). all the parameters,
    besides the data passed, are optional parameters that are default None, and
    each specify a different method of generating a nonlinear feature. one
    passes a list, tuple, or Index of column labels to a named parameter if one
    wishes to perform that kind of feature creation on those particular columns.
    if all named parameters are None, the function simply returns its input.

    see also pair_nlfe.

    notes: for convenience, one should pass an unpacked dictionary of arguments.
           the function will raise KeyError if there are unknown columns passed
           at any step. note that for ln, if there is a math domain error for a
           column, the function will simply skip the column.

    parameters:

    data       a pandas Series or DataFrame, with valid column labels.
    verbose    optional, default False. if True, the verbose output.
    poly2      optional, default None. compute second degree polynomial
               features; i.e. square the value of each column. if passed a list
               or list-like of feature labels, a new feature column "name^2" is
               created for each feature label "name" passed.
    poly3      optional, default None. compute third degree polynomial features;
               i.e. cube the each column. if passed a list or a list-like of
               feature labels, creates for each passed feature label "name" a
               new column "name^3".
    mquad      optional, default None. applies multiquadratic radial basis
               function to features, defined as f(x) = sqrt(1 + x^2). if passed
               a list or list-like of feature labels, for each feature "name",
               creates a new column "name_mq". think of this like poly2 but
               with more linear behavior away from the origin.
    ln         optional, default None. applies natural log to each feature. if
               passed a list or list-like of feature labels creates for each
               feature "name" a new column "name_ln".
    lnquad     optional, default None. applies f(x) = log(1 + x^2) to each
               feature. if passed a list or list-like of feature labels, for
               each feature label "name", creates new column "name_lnq". think
               of this as a symmetric function behaving quadratically near the
               origin but logarithmically away from the origin.
    """
    # save function name
    _fname = nlfe.__name__
    # require that the data be a pandas DataFrame
    if not isinstance(data, DataFrame):
        raise TypeError("{0}: data must be a pandas DataFrame".format(_fname))
    # for second DataFrame to hold new feature columns
    new_cols = None
    # note: technically you could pass a dict, but something would break later
    # if poly2 is not None
    if poly2 is not None:
        # check if argument is iterable and if not, raise TypeError, and also
        # check that all column names are in data frame. any unknown columns
        # are sent to stderr and a KeyError is raised.
        _check_dfcols(data, poly2, _fname, "poly2")
        # for each column name specified in poly2, compute new polynomial
        # feature and assign new column names for poly2_cols
        poly2_cols = power(data[poly2], 2)
        poly2_cols.columns = [col + "^2" for col in poly2]
        # merge with new_cols if not None, else assign
        if new_cols is None: new_cols = poly2_cols
        else: new_cols = new_cols.join(poly2_cols)
        # if verbose, proudly announce computation of these new columns. use
        # subsequent_indent to maintain the indentation level of output.
        if verbose == True:
            print("\n".join(wrap("{0}: poly2: computed new features {1}"
                                 "".format(_fname, list(poly2_cols.columns)),
                                 width = 80, subsequent_indent = " " *
                                 (9 + len(_fname)))))
    # if poly3 is not None
    if poly3 is not None:
        # check if argument is iterable and if not, raise TypeError, and also
        # check that all column names are in data frame. any unknown columns
        # are sent to stderr and a KeyError is raised.
        _check_dfcols(data, poly3, _fname, "poly3")
        # for each column name specified in poly3, compute new polynomial
        # feature and assign new column names for poly3_cols
        poly3_cols = power(data[poly3], 3)
        poly3_cols.columns = [col + "^3" for col in poly3]
        # merge with new_cols if not None, else assign
        if new_cols is None: new_cols = poly3_cols
        else: new_cols = new_cols.join(poly3_cols)
        # if verbose, proudly announce computation of these new columns. use
        # subsequent_indent to maintain the indentation level of output.
        if verbose == True:
            print("\n".join(wrap("{0}: poly3: computed new features {1}"
                                 "".format(_fname, list(poly3_cols.columns)),
                                 width = 80, subsequent_indent = " " *
                                 (9 + len(_fname)))))
    # if mquad is not None
    if mquad is not None:
        # check if argument is iterable and if not, raise TypeError, and also
        # check that all column names are in data frame. any unknown columns
        # are sent to stderr and a KeyError is raised.
        _check_dfcols(data, mquad, _fname, "mquad")
        # for each column name specified in mquad, compute new multiquadratic
        # feature and assign new column names for mquad_cols
        mquad_cols = DataFrame(map(lambda x: sqrt(1 + power(x, 2)),
                                   data[mquad].values),
                               columns = [col + "_mq" for col in mquad])
        # merge with new_cols if not None, else assign
        if new_cols is None: new_cols = mquad_cols
        else: new_cols = new_cols.join(mquad_cols)
        # if verbose, proudly announce computation of these new columns. use
        # subsequent_indent to maintain the indentation level of output.
        if verbose == True:
            print("\n".join(wrap("{0}: mquad: computed new features {1}"
                                 "".format(_fname, list(mquad_cols.columns)),
                                 width = 80, subsequent_indent = " " *
                                 (9 + len(_fname)))))
    # if ln is not None
    if ln is not None:
        # check if argument is iterable and if not, raise TypeError, and also
        # check that all column names are in data frame. any unknown columns
        # are sent to stderr and a KeyError is raised.
        _check_dfcols(data, ln, _fname, "ln")
        # columns for ln features
        ln_cols = None
        # for each column name specified in ln, compute new natural log feature
        # and assign new column names for ln_cols (we don't use numpy log since
        # it only passes warning and will give us NaN; we want to catch the
        # ValueError from the python log function)
        for col in ln:
            # try to compute natural log feature from data[col]
            lnf = None
            try: lnf = DataFrame(map(log, data[col]), columns = [col])
            # if there is a value error, print custom message and continue
            except ValueError:
                print("{0}: ln: value in column {1} outside domain of log"
                      "".format(_fname, col), file = stderr)
                continue
            # if ln_cols is None, set to lnf
            if ln_cols is None: ln_cols = lnf
            # else join lnf to ln_cols
            else: ln_cols = ln_cols.join(lnf)
        # if ln_cols is None, print to stderr and do nothing. we use textwrap to
        # wrap the potentially long error message.
        if ln_cols is None:
            print("\n".join(wrap("{0}: ln: columns {1} contain values outside "
                                 "domain of log".format(_fname, list(ln)),
                                 width = 80, subsequent_indent = " " *
                                 (6 + len(_fname)))), file = stderr)
        # else assign columns, merge with new_cols if new_cols is not None
        else:
            ln_cols.columns = [col + "_ln" for col in ln_cols.columns]
            if new_cols is None: new_cols = ln_cols
            else: new_cols = new_cols.join(ln_cols)
            # if verbose, proudly announce computation of these new columns. use
            # subsequent_indent to maintain the indentation level of output.
            if verbose == True:
                print("\n".join(wrap("{0}: ln: computed new features {1}"
                                     "".format(_fname, list(ln_cols.columns)),
                                     width = 80, subsequent_indent = " " *
                                     (6 + len(_fname)))))
    # if lnquad is not None
    if lnquad is not None:
        # check if argument is iterable and if not, raise TypeError, and also
        # check that all column names are in data frame. any unknown columns
        # are sent to stderr and a KeyError is raised.
        _check_dfcols(data, lnquad, _fname, "lnquad")
        # for each column name specified in lnquad, compute new feature and
        # assign new column names for lnquad_cols
        lnquad_cols = DataFrame(map(lambda x:
                                    map(lambda y: log(1 + power(y, 2)), x),
                                    data[lnquad].values),
                                columns = [col + "_lnq" for col in lnquad])
        # merge with new_cols if not None, else assign
        if new_cols is None: new_cols = lnquad_cols
        else: new_cols = new_cols.join(lnquad_cols)
        # if verbose, proudly announce computation of these new columns. use
        # subsequent_indent to maintain the indentation level of output.
        if verbose == True:
            print("\n".join(wrap("{0}: lnquad: computed new features {1}"
                                 "".format(_fname, list(lnquad_cols.columns)),
                                 width = 80, subsequent_indent = " " *
                                 (10 + len(_fname)))))
    # if new_cols is None, then all keyword arguments are None. return data
    if new_cols is None:
        # if verbose, print a warning
        if verbose == True:
            print("{0}: warning: no keyword arguments passed. returning data"
                  "".format(_fname), file = stderr)
        # return data
        return data
    # else join new_cols with data and return; print total number of feature
    # columns created at the end of the function call if verbose is True
    if verbose == True:
        print("{0}: computed {1} new features".format(_fname,
                                                      len(new_cols.columns)))
    return data.join(new_cols)

def _pairprods(data, cols, _fn, _colsn):
    """
    computes pairwise products of columns cols from DataFrame data. if any
    columns in cols are not in the DataFrame data, then a KeyError_ will be
    raised detailing the missing columns. see above for the definition of the
    KeyError_ class. calls _check_dfcols internally, so no need for a separate
    call to _check_dfcols; the _fn parameter is for the calling function's name
    and _colsn is the name of cols as a string. since the parameters are the
    same and used the same as in _check_dfcols, see above for more help.

    note: if cols is length 1, then raise a RuntimeError indicating that you
          need to pass at least 1 column to compute products of column pairs.
    """
    # save function name
    _fname = _pairprods.__name__
    # check if argument is iterable and if not, raise TypeError, and also
    # check that all column names are in data frame. any unknown columns
    # are sent to stderr and a KeyError is raised.
    _check_dfcols(data, cols, _fn, _colsn)
    # check length of cols; if it is 1, then raise RuntimeError
    if len(cols) == 1:
        raise RuntimeError("{0}: {1}: must pass >1 column".format(_fn, _colsn))
    # now that we know all columns in cols are in data, we can compute our
    # pairwise column products. the convention is that all the columns will be
    # named "name1_name2" for any pair of columns "name1", "name2".
    pair_cols = None
    # do double loop through cols; first get length
    ncols = len(cols)
    for i in range(ncols - 1):
        for j in range(i + 1, ncols):
            # compute the product of data[cols[i]] and data[cols[j]]
            pair = DataFrame(data[cols[i]] * data[cols[j]],
                             columns = [cols[i] + "_" + cols[j]])
            # if pair_cols is None, simply set
            if pair_cols is None: pair_cols = pair
            # else just join with pair_cols
            else: pair_cols = pair_cols.join(pair)
    # return pair_cols
    return pair_cols

def pair_nlfe(data, verbose = False, poly1 = None, poly2 = None, poly3 = None,
              mquad = None, ln = None, lnquad = None):
    """
    given a pandas Series or DataFrame of continuous data, create various
    nonlinear features from pairwise products of features. all parameters,
    besides the data passed, are optional parameters that are default None, and
    each specify a different function to apply to pairwise products of features.
    one passes a list or list-like of column labels to each named argument if
    one wishes to compute the pairwise products between those columns and apply
    the given function to the computed products. if all the named parameters are
    None, simply returns the input data.

    see also nlfe.

    note: for convenience, one should pass an unpacked dictionary of arguments.
          remember that each transformation listed below operates on a pairwise
          product of features, not a single feature as in nlfe. as in nlfe, a
          KeyError will be raised at any step if there are unknown columns + the
          computation of natural log of the pairwise feature products will just
          skip any products that contain values outside of the log domain.

          one should be aware that running pair_nlfe with multiple arguments
          will cause a feature explosion. for n features passed to a named arg,
          multiplying them pairwise will cause the creation of n(n - 1) / 2 new
          feature columns. please use as few options/columns as possible.

    parameters:

    data       a pandas Series or DataFrame, with valid column labels.
    verbose    optional, default False. if True, the verbose output.
    poly1      optional, default None. computes feature products only without
               applying any transformation to the products. if passed a list or
               list-like of feature labels, for each pair of feature labels
               "name1", "name2", creates a new feature "name1_name2".

               note: that is a 1 (one), not an l (the letter L)! looks the same.

    poly2      optional, default None. compute second degree polynomial
               features; i.e. square the value of each column. if passed a list
               or list-like of feature labels, for each pair of feature labels
               "name1", "name2", creates a new feature "name1^2_name2^2".
    poly3      optional, default None. compute third degree polynomial features;
               i.e. cube the each column. if passed a list or a list-like of
               feature labels, for each pair of feature labels "name1", "name2",
               creates a new feature "name1^3_name2^3".
    mquad      optional, default None. applies multiquadratic radial basis
               function to features, defined as f(x) = sqrt(1 + x^2). if passed
               a list or list-like of feature labels, for each pair of feature
               labels "name1", "name2", creates a new feature "name1_name2_mq".
               think of the multiquadratic function being like poly2 but with
               more linear behavior away from the origin.
    ln         optional, default None. applies natural log. if passed a list or
               list-like of feature labels, for each pair of feature labels
               "name1", "name2", creates a new feature column "name1_name2_ln".
    lnquad     optional, default None. applies f(x) = log(1 + x^2) to each
               feature. if passed a list or list-like of feature labels, for
               each pair of feature labels "name1", "name2", creates new feature
               "name1_name2_lnq". think of this as a symmetric function behaving
               quadratically near the origin but logarithmically away from it.
    """
    # save function name
    _fname = pair_nlfe.__name__
    # require that the data be a pandas DataFrame
    if not isinstance(data, DataFrame):
        raise TypeError("{0}: data must be a pandas DataFrame".format(_fname))
    # note: technically you could pass a dict, but something would break later
    # for second DataFrame to hold new pair product columns
    pair_cols = None
    # if poly1 is not None
    if poly1 is not None:
        # runs _check_dfcols internally to check if all columns in poly1 are
        # actually in data and the other checks. then also computes the pairwise
        # products of all the columns.
        poly1_cols = _pairprods(data, poly1, _fname, "poly1")
        # since the naming convention is already set, merge with new_cols if not
        # None, else assign as usual without changing column names
        if pair_cols is None: pair_cols = poly1_cols
        else: pair_cols = pair_cols.join(poly1_cols)
        # if verbose, proudly announce computation of these new columns. use
        # subsequent_indent to maintain the indentation level of output.
        if verbose == True:
            print("\n".join(wrap("{0}: poly1: computed new features {1}"
                                 "".format(_fname, list(poly1_cols.columns)),
                                 width = 80, subsequent_indent = " " *
                                 (9 + len(_fname)))))
    # if poly2 is not None
    if poly2 is not None:
        # runs _check_dfcols internally to check if all columns in poly2 are
        # actually in data and the other checks. then also computes the pairwise
        # products of all the columns.
        poly2_cols = _pairprods(data, poly2, _fname, "poly2")
        # for each column name specified in poly2, compute new polynomial
        # feature and assign new column names for poly2_cols
        poly2_cols = power(poly2_cols, 2)
        poly2_cols.columns = ["^2_".join(col.split("_")) + "^2" for col in
                              poly2_cols.columns]
        # merge with pair_cols if not None, else assign
        if pair_cols is None: pair_cols = poly2_cols
        else: pair_cols = pair_cols.join(poly2_cols)
        # if verbose, proudly announce computation of these new columns. use
        # subsequent_indent to maintain the indentation level of output.
        if verbose == True:
            print("\n".join(wrap("{0}: poly2: computed new features {1}"
                                 "".format(_fname, list(poly2_cols.columns)),
                                 width = 80, subsequent_indent = " " *
                                 (9 + len(_fname)))))
    # if poly3 is not None:
    if poly3 is not None:
        # runs _check_dfcols internally to check if all columns in poly3 are
        # actually in data and the other checks. then also computes the pairwise
        # products of all the columns.
        poly3_cols = _pairprods(data, poly3, _fname, "poly3")
        # for each column name specified in poly3, compute new polynomial
        # feature and assign new column names for poly3_cols
        poly3_cols = power(poly3_cols, 3)
        poly3_cols.columns = ["^3_".join(col.split("_")) + "^3" for col in
                              poly3_cols.columns]
        # merge with pair_cols if not None, else assign
        if pair_cols is None: pair_cols = poly3_cols
        else: pair_cols = pair_cols.join(poly3_cols)
        # if verbose, proudly announce computation of these new columns. use
        # subsequent_indent to maintain the indentation level of output.
        if verbose == True:
            print("\n".join(wrap("{0}: poly3: computed new features {1}"
                                 "".format(_fname, list(poly3_cols.columns)),
                                 width = 80, subsequent_indent = " " *
                                 (9 + len(_fname)))))
    # if mquad is not None:
    if mquad is not None:
        # runs _check_dfcols internally to check if all columns in mquad are
        # actually in data and the other checks. then also computes the pairwise
        # products of all the columns.
        mquad_cols = _pairprods(data, mquad, _fname, "mquad")
        # for each column name specified in mquad, compute new polynomial
        # feature and assign new column names for mquad_cols
        mquad_cols = DataFrame(map(lambda x: sqrt(1 + power(x, 2)),
                                   mquad_cols.values),
                               columns = [col + "_mq" for col in
                                          mquad_cols.columns])
        # merge with pair_cols if not None, else assign
        if pair_cols is None: pair_cols = mquad_cols
        else: pair_cols = pair_cols.join(mquad_cols)
        # if verbose, proudly announce computation of these new columns. use
        # subsequent_indent to maintain the indentation level of output.
        if verbose == True:
            print("\n".join(wrap("{0}: mquad: computed new features {1}"
                                 "".format(_fname, list(mquad_cols.columns)),
                                 width = 80, subsequent_indent = " " *
                                 (9 + len(_fname)))))
    # if ln is not None:
    if ln is not None:
        # runs _check_dfcols internally to check if all columns in ln are
        # actually in data and the other checks. then also computes the pairwise
        # products of all the columns.
        ln_cols = _pairprods(data, ln, _fname, "ln")
        # cols is a list of column names for ln_cols; note that some columns can
        # have log applied to them; maybe not all. so we make a list of column
        # names that we append to as we go; any column that log can be applied
        # to and can be replaced we will add a "_ln" suffix to its column name
        # in the list and will have this modified name added to the list
        # mod_cols. at the very end, we will select just the columns with labels
        # in mod_cols to be appended to pair_cols. by only remembering the names
        # of the modified columns, which are then used to replace the column in
        # ln_cols they were calculated from, and then indexing just those
        # modified columns, we save a lot of memory.
        cols, mod_cols = [], []
        # for each column in ln_cols, try to compute new natural log feature
        # and assign new column names for ln_cols (we don't use numpy log since
        # it only passes warning and will give us NaN; we want to catch the
        # ValueError from the python log function. same as in nlfe.
        for col in ln_cols.columns:
            # first append col to cols
            cols.append(col)
            # try to compute natural log feature from ln_cols; column name is
            # not critical since when we replace a column in ln_cols we won't
            # change the column name (we do that in another step)
            lnf = None
            try: lnf = DataFrame(map(log, ln_cols[col]), columns = [col])
            # if there is a value error, print custom message and continue
            except ValueError:
                print("{0}: ln: value in column {1} outside domain of log"
                      "".format(_fname, col), file = stderr)
                continue
            # if successful, replace ln_cols[col] with lnf, modify last element
            # of cols (just appended col) to col + "_ln", which we also append
            # to mod_cols.
            ln_cols[col] = lnf
            cols[-1] = cols[-1] + "_ln"
            mod_cols.append(cols[-1])
        # if mod_cols has length 0, we were not able to use log on any column.
        # so print to stderr and do nothing. we use textwrap to wrap the
        # potentially long error message.
        if len(mod_cols) == 0:
            print("\n".join(wrap("{0}: ln: all computed columns {1} contain "
                                 "values outside domain of log"
                                 "".format(_fname, list(ln_cols.columns)),
                                 width = 80, subsequent_indent = " " *
                                 (6 + len(_fname)))), file = stderr)
        # else modify columns of ln_cols, select the modified columns, and merge
        # with pair_cols if pair_cols is not None
        else:
            ln_cols.columns = cols
            ln_cols = ln_cols[mod_cols]
            if pair_cols is None: pair_cols = ln_cols
            else: pair_cols = pair_cols.join(ln_cols)
            # if verbose, proudly announce computation of these new columns. use
            # subsequent_indent to maintain the indentation level of output.
            if verbose == True:
                print("\n".join(wrap("{0}: ln: computed new features {1}"
                                     "".format(_fname, list(ln_cols.columns)),
                                     width = 80, subsequent_indent = " " *
                                     (6 + len(_fname)))))
    # if lnquad is not None:
    if lnquad is not None:
        # runs _check_dfcols internally to check if all columns in lnquad are
        # actually in data and the other checks. then also computes the pairwise
        # products of all the columns.
        lnquad_cols = _pairprods(data, lnquad, _fname, "lnquad")
        # for each column name specified in lnquad, compute new polynomial
        # feature and assign new column names for lnquad_cols
        lnquad_cols = DataFrame(map(lambda x: sqrt(1 + power(x, 2)),
                                   lnquad_cols.values),
                               columns = [col + "_lnq" for col in
                                          lnquad_cols.columns])
        # merge with pair_cols if not None, else assign
        if pair_cols is None: pair_cols = lnquad_cols
        else: pair_cols = pair_cols.join(lnquad_cols)
        # if verbose, proudly announce computation of these new columns. use
        # subsequent_indent to maintain the indentation level of output.
        if verbose == True:
            print("\n".join(wrap("{0}: lnquad: computed new features {1}"
                                 "".format(_fname, list(lnquad_cols.columns)),
                                 width = 80, subsequent_indent = " " *
                                 (10 + len(_fname)))))
    # if pair_cols is None, then all keyword arguments are None. return data
    if pair_cols is None:
        # if verbose, print a warning
        if verbose == True:
            print("{0}: warning: no keyword arguments passed. returning data"
                  "".format(_fname), file = stderr)
        # return data
        return data
    # else join new_cols with data and return; print total number of feature
    # columns created at the end of the function call if verbose is True
    if verbose == True:
        print("{0}: computed {1} new features".format(_fname,
                                                      len(pair_cols.columns)))
    return data.join(pair_cols)

if __name__ == "__main__":
    print("{0}: do not run module as script. refer to docstring for usage."
          "".format(_MODULE_NAME), file = stderr)
