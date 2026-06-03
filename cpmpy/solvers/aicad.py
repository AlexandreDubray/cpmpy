#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## aicad.py
##
"""
    Interface to aicad's API

    .. note::
        [GUIDELINE] Replace <TEMPLATE> by the solver's name, and implement the missing pieces
        The functions are ordered in a way that could be convenient to 
        start from the top and continue in that order.

    .. note::
        After you are done filling in the template, remove all comments starting with [GUIDELINE]

    .. warning::
        [GUIDELINE] do not include the python package at the top of the file,
        as CPMpy should also work without this solver installed.
        To ensure that, include it inside supported() and other functions that need it...

    <some information on the solver>

    Always use :func:`cp.SolverLookup.get("aicad") <cpmpy.solvers.utils.SolverLookup.get>` to instantiate the solver object.

    ============
    Installation
    ============

    Requires that the 'pyaicad' python package is installed:

    .. code-block:: console
    
        $ pip install pyaicad

    See detailed installation instructions at:
    https://github.com/AlexandreDubray/aicad

    The rest of this documentation is for advanced users.

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_aicad
"""

from typing import Optional
import warnings
from packaging.version import Version
import time

from .solver_interface import SolverInterface, SolverStatus, ExitStatus, Callback
from ..expressions.core import Expression, Comparison, Operator
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl, _NumVarImpl
from ..expressions.utils import is_num, is_any_list, is_boolexpr, is_int
from ..transformations.get_variables import get_variables
from ..transformations.normalize import toplevel_list
from ..transformations.safening import no_partial_functions
from ..transformations.decompose_global import decompose_in_tree, decompose_objective
from ..transformations.flatten_model import flatten_constraint, flatten_objective
from ..transformations.comparison import only_numexpr_equality
from ..transformations.reification import reify_rewrite, only_bv_reifies
from ..transformations.safening import safen_objective


class CPM_aicad(SolverInterface):
    """
    Interface to aicad's API

    Creates the following attributes (see parent constructor for more):
    - tpl_model: object, aicad's model object

    Documentation of the solver's own Python API:
    https://github.com/AlexandreDubray/aicad/tree/main/python
    """

    # [GUIDELINE] list all supported global constraints and global functions
    #           (e.g., 'alldifferent', 'max', 'element', ...)
    supported_global_constraints = frozenset({'alldifferent'})
    # [GUIDELINE] list all global constraints supported in reified context (or half-reified if transformed)
    #           (e.g., 'alldifferent' if your solver supports `b -> AllDifferent(X)`)
    supported_reified_global_constraints = frozenset()

    @staticmethod
    def supported():
        # try to import the package
        try:
            import pyaicad as gp
            # optionally enforce a specific version
            aicad_version = CPM_aicad.version()
            if Version(aicad_version) < Version("0.0.1"):
                warnings.warn(f"CPMpy uses features only available from pyaicad version 0.0.1, "
                              f"but you have version {aicad_version}.")
                return False
            return True
        except ModuleNotFoundError: # if solver's Python package is not installed
            return False

    @classmethod
    def version(cls) -> Optional[str]:
        """
        Returns the installed version of the solver's Python API.
        """
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version('pyaicad')
        except PackageNotFoundError:
            return None

    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Arguments:
        - cpm_model: Model(), a CPMpy Model() (optional)
        - subsolver: str, name of a subsolver (optional)
        """
        if not self.supported():
            raise ModuleNotFoundError("CPM_aicad: Install the python package 'cpmpy[aicad]' to use this solver interface.")   

        import pyaicad 

        assert subsolver is None # unless you support subsolvers, see pysat or minizinc

        # initialise the native solver object
        self.acd_solver = pyaicad.Solver()

        # initialise everything else and post the constraints/objective
        super().__init__(name="TEMPLATE", cpm_model=cpm_model)


    @property
    def native_model(self):
        """
            Returns the solver's underlying native model (for direct solver access).
        """
        return None

    def solve(self, time_limit:Optional[float]=None, **kwargs):
        """
            Call the TEMPLATE solver

            Arguments:
            - time_limit:  maximum solve time in seconds (float, optional)
            - kwargs:      any keyword argument, sets parameters of solver object

            Arguments that correspond to solver parameters:
            - ``max_width`` : Optional[int]
            - ``order``: Optional[pyaicad::PyOrderingHeuristic]
            - ``merge``: Optional[pyaicad::PyMergeHeuristic]

            For a complete description of the parameters, see https://github.com/AlexandreDubray/aicad
        """

        from pyaicad import PyOrderingHeuristic, PyMergeHeuristic

        # ensure all vars are known to solver
        start = time.time()
        self.solver_vars(list(self.user_vars))
        end = time.time()

        max_width = kwargs.get("max_width")
        if max_width is None:
            max_width = 2**64 - 1

        order = kwargs.get("order")
        if order is None:
            order = PyOrderingHeuristic.MinDomMaxLinked()

        merge = kwargs.get("merge")
        if merge is None:
            merge = PyMergeHeuristic.LessRelaxed

        # TODO: Handle time limit
        sol = self.acd_solver.solve(max_width, order, merge)

        # [GUIDELINE] consider saving the status as self.TPL_status so that advanced CPMpy users can access the status object.
        #       This is mainly useful when more elaborate information about the solve-call is saved into the status

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = end - start

        # Translate solver exit status to CPMpy exit status
        # CSP:                         COP:
        # ├─ sat -> FEASIBLE           ├─ optimal -> OPTIMAL
        # ├─ unsat -> UNSATISFIABLE    ├─ sub-optimal -> FEASIBLE
        # └─ timeout -> UNKNOWN        ├─ unsat -> UNSATISFIABLE
        #                              └─ timeout -> UNKNOWN
        if self.acd_solver.is_unsat():
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif sol is not None:
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        else:
            self.cpm_status.exitstatus = ExitStatus.UNKNOWN

        # True/False depending on self.cpm_status
        has_sol = self._solve_return(self.cpm_status)

        # translate solution values (of user specified variables only)
        self.objective_value_ = None
        if has_sol:
            # fill in variable values
            for cpm_var in self.user_vars:
                sol_var = self.solver_var(cpm_var)
                cpm_var._value = sol[sol_var]

            # translate objective, for optimisation problems only
            if self.has_objective():
                raise NotImplementedError("Optimisation is not supported by aicad")

        else: # clear values of variables
            for cpm_var in self.user_vars:
                cpm_var.clear()

        return has_sol


    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created
        """
        if is_num(cpm_var): # shortcut, eases posting constraints
            if not is_int(cpm_var):
                raise ValueError(f"Aicad only accepts integer constants, got {cpm_var} of type{type(cpm_var)}")
            return int(cpm_var)

        # [GUIDELINE] some solver interfaces explicitely create variables on a solver object
        #       then use self.TPL_solver.NewBoolVar(...) instead of TEMPLATEpy.NewBoolVar(...)

        # special case, negative-bool-view
        # work directly on var inside the view
        if isinstance(cpm_var, NegBoolView):
            return self.acd_solver.negate(self.solver_var(cpm_var._bv))

        # create if it does not exist
        if cpm_var.name not in self._varmap:
            if isinstance(cpm_var, _BoolVarImpl):
                revar = self.acd_solver.add_bool_var()
            elif isinstance(cpm_var, _IntVarImpl):
                revar = self.acd_solver.add_int_var([x for x in range(cpm_var.lb, cpm_var.ub + 1)])
            else:
                raise NotImplementedError("Not a known var {}".format(cpm_var))
            self._varmap[cpm_var.name] = revar

        # return from cache
        return self._varmap[cpm_var.name]

    def has_objective(self):
        return False

    def _make_numexpr(self, cpm_expr):
        """
            Converts a numeric CPMpy 'flat' expression into a solver-specific numeric expression

            Primarily used for setting objective functions, and optionally in constraint posting
        """

        # [GUIDELINE] not all solver interfaces have a native "numerical expression" object.
        #       in that case, this function may be removed and a case-by-case analysis of the numerical expression
        #           used in the constraint at hand is required in `add()`
        #       For an example of such solver interface, check out solvers/choco.py or solvers/exact.py

        if is_num(cpm_expr):
            return cpm_expr

        # decision variables, check in varmap
        if isinstance(cpm_expr, _NumVarImpl):  # _BoolVarImpl is subclass of _NumVarImpl
            return self.solver_var(cpm_expr)

        # any solver-native numerical expression
        if isinstance(cpm_expr, Operator):
           if cpm_expr.name == 'sum':
               return self.TPL_solver.sum(self.solver_vars(cpm_expr.args))
           elif cpm_expr.name == 'wsum':
               weights, vars = cpm_expr.args
               return self.TPL_solver.weighted_sum(weights, self.solver_vars(vars))
           # [GUIDELINE] or more fancy ones such as max
           #        be aware this is not the Maximum CONSTRAINT, but rather the Maximum NUMERICAL EXPRESSION
           elif cpm_expr.name == "max":
               return self.TPL_solver.maximum_of_vars(self.solver_vars(cpm_expr.args))
           # ...
        raise NotImplementedError("Aicad: Not a known supported numexpr {}".format(cpm_expr))

    # `add()` first calls `transform()`
    def transform(self, cpm_expr):
        """
            Transform arbitrary CPMpy expressions to constraints the solver supports

            Implemented through chaining multiple solver-independent **transformation functions** from
            the `cpmpy/transformations/` directory.

            See the 'Adding a new solver' docs on readthedocs for more information.

        :param cpm_expr: CPMpy expression, or list thereof
        :type cpm_expr: Expression or list of Expression

        :return: list of Expression
        """
        # apply transformations
        # XXX chose the transformations your solver needs, see cpmpy/transformations/
        cpm_cons = toplevel_list(cpm_expr)
        cpm_cons = no_partial_functions(cpm_cons)  # to also safen at toplevel, add: `, safen_toplevel={"element", "div", "mod"})`
        cpm_cons = decompose_in_tree(cpm_cons,
                                     supported=self.supported_global_constraints,
                                     supported_reified=self.supported_reified_global_constraints,
                                     csemap=self._csemap)
        cpm_cons = flatten_constraint(cpm_cons)  # flat normal form
        cpm_cons = reify_rewrite(cpm_cons, supported=frozenset(['sum', 'wsum']))  # constraints that support reification
        cpm_cons = only_bv_reifies(cpm_cons)
        cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum", "sub"]))  # supports >, <, !=
        return cpm_cons

    def add(self, cpm_expr_orig):
        """
            Eagerly add a constraint to the underlying solver.

            Any CPMpy expression given is immediately transformed (through `transform()`)
            and then posted to the solver in this function.

            This can raise 'NotImplementedError' for any constraint not supported after transformation

            The variables used in expressions given to add are stored as 'user variables'. Those are the only ones
            the user knows and cares about (and will be populated with a value after solve). All other variables
            are auxiliary variables created by transformations.

        :param cpm_expr: CPMpy expression, or list thereof
        :type cpm_expr: Expression or list of Expression

        :return: self
        """

        # add new user vars to the set
        get_variables(cpm_expr_orig, collect=self.user_vars)

        # transform and post the constraints
        for cpm_expr in self.transform(cpm_expr_orig):
            if isinstance(cpm_expr, _BoolVarImpl):
                # base case, just var or ~var
                self.TPL_solver.add_clause([ self.solver_var(cpm_expr) ])

            elif isinstance(cpm_expr, Operator):
                if cpm_expr.name == "or":
                    raise NotImplementedError("Aicad: no support (yet) for clauses:", cpm_expr)
                elif cpm_expr.name == "->": # half-reification
                    raise NotImplementedError("Aicaid: no support (yet) for half-reified constraint:", cpm_expr)

            elif isinstance(cpm_expr, Comparison):
                lhs, rhs = cpm_expr.args
                if cpm_expr.name == "==":
                    self.acd_solver.add_equal(self.solver_var(lhs), self.solver_var(rhs))
                else:
                    raise NotImplementedError("Aicad: no support (yet) for comparisons:", cpm_expr)
            # global constraints
            elif cpm_expr.name == "alldifferent":
                self.acd_solver.add_all_different(self.solver_vars(cpm_expr.args))
            else:
                raise NotImplementedError("Aicad: constraint not (yet) supported", cpm_expr)

        return self
    __add__ = add  # avoid redirect in superclass

    # Other functions from SolverInterface that you can overwrite:
    # solveAll, solution_hint, get_core

    def solveAll(self, display:Optional[Callback]=None, time_limit:Optional[float]=None, solution_limit:Optional[int]=None, call_from_model=False, **kwargs):
        """
            A shorthand to (efficiently) compute all (optimal) solutions, map them to CPMpy and optionally display the solutions.

            If the problem is an optimization problem, returns only optimal solutions.

            Arguments:
                - display: either a list of CPMpy expressions, OR a callback function, called with the variables after value-mapping
                        default/None: nothing displayed
                - time_limit: stop after this many seconds (default: None)
                - solution_limit: stop after this many solutions (default: None)
                - call_from_model: whether the method is called from a CPMpy Model instance or not
                - any other keyword argument

            Returns: number of solutions found
        """
        raise NotImplementedError("Aicad does not support yet finding all solutions")

    def solve(self, time_limit:Optional[float]=None, **kwargs):
        """
            Call the TEMPLATE solver

            Arguments:
            - time_limit:  maximum solve time in seconds (float, optional)
            - kwargs:      any keyword argument, sets parameters of solver object

            Arguments that correspond to solver parameters:
            - ``max_width`` : Optional[int]
            - ``order``: Optional[pyaicad::PyOrderingHeuristic]
            - ``merge``: Optional[pyaicad::PyMergeHeuristic]
            - ``force_recompile``: Optional[bool]

            For a complete description of the parameters, see https://github.com/AlexandreDubray/aicad
        """

        from pyaicad import PyOrderingHeuristic, PyMergeHeuristic

        # ensure all vars are known to solver
        start = time.time()
        self.solver_vars(list(self.user_vars))

        # TODO: Handle time limit
        sol = self.acd_solver.solve(**kwargs)
        end = time.time()

        # [GUIDELINE] consider saving the status as self.TPL_status so that advanced CPMpy users can access the status object.
        #       This is mainly useful when more elaborate information about the solve-call is saved into the status

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = end - start

        # Translate solver exit status to CPMpy exit status
        # CSP:                         COP:
        # ├─ sat -> FEASIBLE           ├─ optimal -> OPTIMAL
        # ├─ unsat -> UNSATISFIABLE    ├─ sub-optimal -> FEASIBLE
        # └─ timeout -> UNKNOWN        ├─ unsat -> UNSATISFIABLE
        #                              └─ timeout -> UNKNOWN
        if self.acd_solver.is_unsat():
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif sol is not None:
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        else:
            self.cpm_status.exitstatus = ExitStatus.UNKNOWN

        # True/False depending on self.cpm_status
        has_sol = self._solve_return(self.cpm_status)

        # translate solution values (of user specified variables only)
        self.objective_value_ = None
        if has_sol:
            # fill in variable values
            for cpm_var in self.user_vars:
                sol_var = self.solver_var(cpm_var)
                cpm_var._value = sol[sol_var]

            # translate objective, for optimisation problems only
            if self.has_objective():
                raise NotImplementedError("Optimisation is not supported by aicad")

        else: # clear values of variables
            for cpm_var in self.user_vars:
                cpm_var.clear()

        return has_sol

    def native_solver(self):
        return self.acd_solver
