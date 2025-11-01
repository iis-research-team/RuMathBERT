import csv
import json
import matplotlib.pyplot as plt
import networkx as nx
import pygraphviz as pgv
import random
import sympy as sp
from collections import defaultdict
from tqdm import tqdm
from latex2sympy2_extended import latex2sympy  # pip install latex2sympy2_extended[antlr4_13_2]


class FormulaTree:
    def __init__(self, latex):
        self.edges = []
        self.node_types = []
        self.node_values = []
        self.node_descriptions = []
        self.node_id_counter = 0
        self.latex_string = latex

    def _create_node(self, node_type, value, description):
        node_id = self.node_id_counter
        self.node_types.append(node_type)
        self.node_values.append(value)
        self.node_descriptions.append(description)
        self.node_id_counter += 1
        return node_id

    def _add_edge(self, source, target):
        self.edges.append([source, target])

    def parse_latex(self):
        try:
            sympy_expr = latex2sympy(self.latex_string)
        except Exception as e:
            raise ValueError(f"Parser failure: {str(e)}")

        if sympy_expr is None:
            raise ValueError(f"LaTeX parsed successfully but returned None: {self.latex_string}")

        return self._build_tree(sympy_expr)

    def _build_tree(self, expr):
        # matrix operations - detected first since for matrix operations is_Matrix = True in sympy
        if hasattr(expr, '__class__') and hasattr(expr.__class__, '__name__'):
            class_name = expr.__class__.__name__
            if class_name == 'MatMul':
                return self._handle_multiplication(expr)
            elif class_name == 'MatAdd':
                return self._handle_addition(expr)
            elif class_name == 'MatPow':
                return self._handle_power(expr)

        # matrix detection
        if hasattr(expr, 'is_Matrix') and expr.is_Matrix:
            return self._handle_matrix(expr)

        # atomic expressions
        if hasattr(expr, 'is_Atom') and expr.is_Atom:
            if expr.is_Symbol:
                return self._create_node("Variable", str(expr), f"Variable: {expr}")
            elif expr == sp.E:
                return self._create_node("Constant", "e", "Euler's number")
            elif expr == sp.pi:
                return self._create_node("Constant", "Pi", "Pi constant")
            elif expr == sp.I:
                return self._create_node("Constant", "i", "Imaginary unit")
            elif expr == sp.oo:
                return self._create_node("Constant", "infinity", "Infinity")
            elif expr == sp.zoo:
                return self._create_node("Constant", "complex_infinity", "Complex Infinity")
            elif expr.is_Number:
                return self._create_node("Number", str(expr), f"Number: {expr}")
            else:
                raise ValueError(f"Unsupported atomic expression type: {type(expr)} - {expr}")

        # general operations
        return self._handle_operation(expr)

    def _handle_operation(self, expr):
        if hasattr(expr, 'is_Add') and expr.is_Add:
            return self._handle_addition(expr)
        elif hasattr(expr, 'is_Mul') and expr.is_Mul:
            return self._handle_multiplication(expr)
        elif hasattr(expr, 'is_Pow') and expr.is_Pow:
            return self._handle_power(expr)

        func_name = type(expr).__name__

        if hasattr(expr, 'func') and hasattr(expr, 'args'):
            # container
            if func_name == 'Tuple':
                return self._handle_tuple(expr)

            if hasattr(expr.func, '__name__'):
                func_name_str = expr.func.__name__

                # bounded operations
                if func_name_str in ['Sum', 'Product', 'Integral', 'Limit']:
                    return self._handle_bounded_operation(expr, func_name_str)

                # specific operations
                elif func_name_str == 'Derivative':
                    return self._handle_derivative(expr)
                elif func_name_str == 'log':
                    return self._handle_logarithm(expr)
                elif func_name_str == 'Interval':
                    return self._handle_interval(expr)

                # common functions
                elif func_name_str in ['sin', 'cos', 'tan', 'cot', 'sec', 'csc',
                                       'asin', 'acos', 'atan', 'acot', 'asec', 'acsc',
                                       'sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch',
                                       'asinh', 'acosh', 'atanh', 'acoth', 'asech', 'acsc',
                                       'exp', 'factorial', 'gamma', 'zeta', 'erf',
                                       'And', 'Or', 'Not', 'Xor', 'Nand', 'Nor', 'Xnor',
                                       'Equality', 'Unequality', 'StrictLessThan', 'LessThan',
                                       'StrictGreaterThan', 'GreaterThan',
                                       'Min', 'Max', 'Abs', 'Floor', 'Ceiling', 'Re', 'Im']:
                    root_id = self._create_node("Function", func_name_str, f"{func_name_str} function")
                    for arg in expr.args:
                        arg_id = self._build_tree(arg)
                        self._add_edge(root_id, arg_id)
                    return root_id

            # custom and undocumented functions
            root_id = self._create_node("Function", func_name, f"Function: {func_name}")
            for arg in expr.args:
                arg_id = self._build_tree(arg)
                self._add_edge(root_id, arg_id)
            return root_id

        # undocumented operations
        if hasattr(expr, 'args'):
            root_id = self._create_node("Operation", func_name, f"Operation: {func_name}")
            for arg in expr.args:
                arg_id = self._build_tree(arg)
                self._add_edge(root_id, arg_id)
            return root_id

        raise ValueError(f"No structure to process: {type(expr)} - {expr}")

    def _handle_matrix(self, expr):
        rows, cols = expr.shape

        if rows == 1 and cols == 1:
            return self._build_tree(expr[0])

        # placeholder node values, up for discussion
        if rows == 1:
            matrix_type = "RowVector"
            display_value = f"row[{cols}]"
        elif cols == 1:
            matrix_type = "ColumnVector"
            display_value = f"column[{rows}]"
        else:
            matrix_type = "Matrix"
            display_value = f"matrix[{rows}x{cols}]"

        root_id = self._create_node(matrix_type, display_value, f"{rows}x{cols} matrix")

        if rows > 1 and cols > 1:
            for i in range(rows):
                row_id = self._create_node("MatrixRow", f"row{i + 1}", f"Row {i + 1}")
                self._add_edge(root_id, row_id)

                for j in range(cols):
                    element_id = self._build_tree(expr[i, j])
                    self._add_edge(row_id, element_id)
        else:
            for i in range(rows):
                for j in range(cols):
                    element_id = self._build_tree(expr[i, j])
                    self._add_edge(root_id, element_id)

        return root_id

    def _handle_addition(self, expr):
        args = expr.as_ordered_terms()

        negative_terms = []
        positive_terms = []

        for arg in args:
            coeff, rest = arg.as_coeff_Mul()
            if coeff < 0:
                negative_terms.append((-coeff) * rest if coeff != -1 else rest)
            else:
                positive_terms.append(arg)

        if positive_terms and negative_terms:
            current_root = self._build_tree(positive_terms[0])

            for pos_term in positive_terms[1:]:
                op_id = self._create_node("Operation", "Add", "Addition")
                arg_tree = self._build_tree(pos_term)
                self._add_edge(op_id, current_root)
                self._add_edge(op_id, arg_tree)
                current_root = op_id

            for neg_term in negative_terms:
                op_id = self._create_node("Operation", "Sub", "Subtraction")
                arg_tree = self._build_tree(neg_term)
                self._add_edge(op_id, current_root)
                self._add_edge(op_id, arg_tree)
                current_root = op_id

            return current_root

        elif negative_terms and not positive_terms:
            current_root = self._create_node("Operation", "Mul", "Multiplication")
            neg_one_id = self._create_node("Number", "-1", "Negative one")
            first_neg_tree = self._build_tree(negative_terms[0])
            self._add_edge(current_root, neg_one_id)
            self._add_edge(current_root, first_neg_tree)

            for neg_term in negative_terms[1:]:
                op_id = self._create_node("Operation", "Sub", "Subtraction")
                arg_tree = self._build_tree(neg_term)
                self._add_edge(op_id, current_root)
                self._add_edge(op_id, arg_tree)
                current_root = op_id

            return current_root

        else:
            current_root = self._build_tree(positive_terms[0])
            for pos_term in positive_terms[1:]:
                op_id = self._create_node("Operation", "Add", "Addition")
                arg_tree = self._build_tree(pos_term)
                self._add_edge(op_id, current_root)
                self._add_edge(op_id, arg_tree)
                current_root = op_id
            return current_root

    def _handle_multiplication(self, expr):
        args = expr.as_ordered_factors()

        numerator_terms = []
        denominator_terms = []

        for arg in args:
            if arg.is_Pow and arg.args[1] == -1:
                denominator_terms.append(arg.args[0])
            else:
                numerator_terms.append(arg)

        if denominator_terms:
            if len(numerator_terms) == 1:
                numerator_tree = self._build_tree(numerator_terms[0])
            elif numerator_terms:
                numerator_tree = self._create_node("Operation", "Mul", "Multiplication")
                for term in numerator_terms:
                    term_tree = self._build_tree(term)
                    self._add_edge(numerator_tree, term_tree)
            else:
                numerator_tree = self._create_node("Number", "1", "One")

            if len(denominator_terms) == 1:
                denominator_tree = self._build_tree(denominator_terms[0])
            else:
                denominator_tree = self._create_node("Operation", "Mul", "Multiplication")
                for term in denominator_terms:
                    term_tree = self._build_tree(term)
                    self._add_edge(denominator_tree, term_tree)

            division_id = self._create_node("Operation", "Div", "Division")
            self._add_edge(division_id, numerator_tree)
            self._add_edge(division_id, denominator_tree)
            return division_id

        else:
            current_root = self._build_tree(args[0])
            for arg in args[1:]:
                op_id = self._create_node("Operation", "Mul", "Multiplication")
                arg_tree = self._build_tree(arg)
                self._add_edge(op_id, current_root)
                self._add_edge(op_id, arg_tree)
                current_root = op_id
            return current_root

    def _handle_power(self, expr):
        base, exponent = expr.args[0], expr.args[1]

        if (hasattr(exponent, 'is_Rational') and exponent.is_Rational and
                exponent.p == 1 and exponent.q > 1):
            root_degree = exponent.q

            if root_degree == 2:
                root_id = self._create_node("Function", "Sqrt", "Square root")
                radicand_id = self._build_tree(base)
                self._add_edge(root_id, radicand_id)
                return root_id
            else:
                root_id = self._create_node("Function", f"Root", f"n-th root")

                radicand_id = self._build_tree(base)
                self._add_edge(root_id, radicand_id)

                degree_id = self._create_node("RootDegree", str(root_degree), f"Root degree")
                self._add_edge(root_id, degree_id)

                return root_id

        root_id = self._create_node("Function", "Pow", "Power")
        base_id = self._build_tree(base)
        exp_id = self._build_tree(exponent)

        self._add_edge(root_id, base_id)
        self._add_edge(root_id, exp_id)
        return root_id

    def _handle_bounded_operation(self, expr, op_type):
        # placeholder node values
        op_config = {
            'Sum': ("Function", "Sum", "Sum"),
            'Product': ("Function", "Product", "Product"),
            'Integral': ("Function", "Integral", "Integral"),
            'Limit': ("Function", "lim", "Limit")
        }

        node_type, display_value, description = op_config[op_type]
        root_id = self._create_node(node_type, display_value, description)

        if len(expr.args) >= 1:
            expr_id = self._build_tree(expr.args[0])
            self._add_edge(root_id, expr_id)

        if len(expr.args) >= 2:
            if op_type == 'Limit':
                if len(expr.args) >= 3:
                    limit_point_id = self._create_node("Bounds", "LimitPoint", "Limit bounds")
                    self._add_edge(root_id, limit_point_id)

                    var_id = self._build_tree(expr.args[1])
                    self._add_edge(limit_point_id, var_id)

                    point_id = self._build_tree(expr.args[2])
                    self._add_edge(limit_point_id, point_id)

                    if len(expr.args) >= 4:
                        direction = expr.args[3]
                        if direction == '+' or str(direction) == '+':
                            direction_str = "+"
                            direction_desc = "right-hand limit"
                        elif direction == '-' or str(direction) == '-':
                            direction_str = "-"
                            direction_desc = "left-hand limit"
                        else:
                            direction_str = str(direction)
                            direction_desc = f"direction: {direction}"

                        direction_id = self._create_node("Direction", direction_str, direction_desc)
                        self._add_edge(limit_point_id, direction_id)
            else:
                bounds_id = self._build_tree(expr.args[1])
                self._add_edge(root_id, bounds_id)

        return root_id

    def _handle_derivative(self, expr):
        root_id = self._create_node("Function", "derivative", "Derivative")

        if len(expr.args) > 1:
            expr_id = self._build_tree(expr.args[0])
            self._add_edge(root_id, expr_id)

            arg_id = self._build_tree(expr.args[1][0])
            self._add_edge(root_id, arg_id)

        return root_id

    def _handle_logarithm(self, expr):
        root_id = self._create_node("Function", "log", "Logarithm")

        if len(expr.args) >= 1:
            arg_id = self._build_tree(expr.args[0])
            self._add_edge(root_id, arg_id)

        if len(expr.args) >= 2:
            base_id = self._build_tree(expr.args[1])
            self._add_edge(root_id, base_id)
        else:
            base_id = self._build_tree(sp.E)
            self._add_edge(root_id, base_id)

        return root_id

    def _handle_interval(self, expr):
        root_id = self._create_node("Interval", "interval", "Interval")

        left_expr, right_expr, left_bracket, right_bracket = expr.args

        left_bracket_char = "(" if left_bracket else "["
        left_bracket_desc = "Left bracket (exclusive)" if left_bracket else "Left bracket (inclusive)"
        left_bracket_id = self._create_node("Bracket", left_bracket_char, left_bracket_desc)
        self._add_edge(root_id, left_bracket_id)

        left_expr_id = self._build_tree(left_expr)
        self._add_edge(root_id, left_expr_id)

        right_expr_id = self._build_tree(right_expr)
        self._add_edge(root_id, right_expr_id)

        right_bracket_char = ")" if right_bracket else "]"
        right_bracket_desc = "Right bracket (exclusive)" if right_bracket else "Right bracket (inclusive)"
        right_bracket_id = self._create_node("Bracket", right_bracket_char, right_bracket_desc)
        self._add_edge(root_id, right_bracket_id)

        return root_id

    def _handle_tuple(self, expr):
        args = expr.args

        if len(args) == 1 and not (hasattr(args[0], 'is_Integer') and args[0].is_Integer):
            root_id = self._create_node("IntegrationVariable", "d", "Integration variable")
            var_id = self._build_tree(args[0])
            self._add_edge(root_id, var_id)
            return root_id

        if len(args) == 3:
            root_id = self._create_node("Bounds", "bounds", "Bounds")
            for arg in args:
                arg_id = self._build_tree(arg)
                self._add_edge(root_id, arg_id)
            return root_id

        elif len(args) == 2 and hasattr(args[1], 'is_Integer') and args[1].is_Integer:
            root_id = self._create_node("DifferentiationVariable", "d", "Differentiation")

            var_id = self._build_tree(args[0])
            self._add_edge(root_id, var_id)

            order_id = self._build_tree(args[1])
            self._add_edge(root_id, order_id)

            return root_id

        # unknown tuple
        else:
            root_id = self._create_node("Tuple", "tuple", "Tuple")
            for arg in args:
                arg_id = self._build_tree(arg)
                self._add_edge(root_id, arg_id)
            return root_id

    def get_tree_structure(self):
        return {
            "nodes": [
                {
                    "id": i,
                    "type": self.node_types[i],
                    "value": self.node_values[i],
                    "description": self.node_descriptions[i]
                }
                for i in range(self.node_id_counter)
            ],
            "edges": self.edges,
            "latex_string": self.latex_string
        }

    def print_tree(self):
        G = nx.DiGraph()

        for i in range(self.node_id_counter):
            label = f"{self.node_values[i]} ({self.node_types[i]})"
            G.add_node(i, label=label)

        for source, target in self.edges:
            G.add_edge(source, target)

        root_id = min(set(range(self.node_id_counter)) - set(edge[1] for edge in self.edges))

        pos = nx.nx_agraph.graphviz_layout(G, prog='dot', root=root_id)

        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, labels={i: G.nodes[i]['label'] for i in G.nodes()},
                node_color='lightblue', node_size=1500, font_size=8, font_weight='bold',
                arrows=True, arrowstyle='-|>', arrowsize=20)
        plt.title(f"Formula Tree: {self.latex_string}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def generate_opts_from_csv(csv_filepath, output_json_filepath):
    formula_trees_data = []
    success_count = 0
    error_count = 0
    error_categories = defaultdict(int)
    unsupported_items = defaultdict(int)

    with open(csv_filepath, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        rows = list(csv_reader)

        for row in tqdm(rows, desc="Generating OPTs: "):
            if "formula" in row and row["formula"].strip():
                latex_str = row["formula"].strip()

                try:
                    formula_tree = FormulaTree(latex_str)
                    formula_tree.parse_latex()

                    tree_structure = formula_tree.get_tree_structure()
                    if formula_tree.node_id_counter > 1:
                        formula_trees_data.append(tree_structure)
                        success_count += 1
                    else:
                        error_count += 1
                        category = "Single node (no operator in the formula)"

                    del formula_tree
                    error_categories[category] += 1

                except Exception as e:
                    error_count += 1

                    error_msg = str(e)
                    error_type = type(e).__name__

                    if "Unsupported function type" in error_msg:
                        if " - " in error_msg:
                            func_name = error_msg.split(": ")[1].split(" - ")[0]
                            unsupported_items[f"function:{func_name}"] += 1
                        category = "Unsupported Function"

                    elif "Unsupported expression type" in error_msg:
                        if " - " in error_msg:
                            expr_type = error_msg.split(": ")[1].split(" - ")[0]
                            unsupported_items[f"expression:{expr_type}"] += 1
                        category = "Unsupported Expression Type"

                    elif "Unsupported atomic expression type" in error_msg:
                        if " - " in error_msg:
                            atomic_type = error_msg.split(": ")[1].split(" - ")[0]
                            unsupported_items[f"atomic:{atomic_type}"] += 1
                        category = "Unsupported Atomic Type"

                    elif "Failed to parse LaTeX expression" in error_msg:
                        category = "LaTeX Parsing Error"
                    elif "LaTeX parsed successfully but returned Non" in error_msg:
                        category = "LaTeX Parsing returns None"
                    elif "Failed to process matrix" in error_msg:
                        category = "Matrix Processing Error"
                    else:
                        category = f"Other Error: {error_type}"

                    error_categories[category] += 1

    with open(output_json_filepath, 'w', encoding='utf-8') as f:
        json.dump(formula_trees_data, f, indent=2, ensure_ascii=False)

    print(f"Success: {success_count}")
    print(f"Fail: {error_count}")

    if error_categories:
        print("Errors:")
        for category, count in sorted(error_categories.items(), key=lambda x: x[1], reverse=True):
            print(f"  {category}: {count}")


# generate_opts_from_csv("ruwikiformulae.csv", "OPTs.json")

def plot_opts(file_path, sample_size=10):
    with open(file_path, 'r') as file:
        data = json.load(file)
        for formula in random.sample(data, sample_size):
            formula_tree = FormulaTree(formula["latex_string"])
            formula_tree.edges = formula["edges"]
            formula_tree.node_id_counter = len(formula["nodes"])
            for node in formula["nodes"]:
                formula_tree.node_types.append(node["type"])
                formula_tree.node_values.append(node["value"])
                formula_tree.node_descriptions.append(node["description"])

            formula_tree.print_tree()


if __name__ == "__main__":
    plot_opts("OPTs.json")
