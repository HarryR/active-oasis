# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import trstools as T
from py_ecc import bn128
from random import randint
from hashlib import sha256
from py_ecc.bn128 import add, multiply, double, curve_order, field_modulus, G1, eq
from py_ecc.bn128.bn128_field_elements import inv


bytes_to_int = lambda x: reduce(lambda o, b: (o << 8) + ord(b), [0] + list(x))
rands = lambda: randint(1, curve_order - 1)
sbmul = lambda s: multiply(G1, s)
hashs = lambda *x: bytes_to_int(sha256('.'.join(['%X' for _ in range(0, len(x))]) % x).digest()) % curve_order
hashp = lambda *x: hashs(*[item.n for sublist in x for item in sublist])

fake_g = rands()
fake_G = sbmul(fake_g)
fake_sbmul = lambda x: multiply(fake_G, x)


expr_funcs = dict(
	TypeScalar=lambda x: x,
	TypePoint=lambda x: x,
	Opaque=lambda x: x,
	G=fake_sbmul,
	GDL=lambda: fake_g,
	Point=lambda x: x if isinstance(x, tuple) else fake_sbmul(x),
	PointDouble=lambda x: double(x),
	Sub=lambda x, y: (x - y) % curve_order,
	Neg=lambda x: -x % curve_order,
	Inv=lambda x: inv(x, curve_order),
	#InvMul=lambda x, y: (x * pow(y, field_modulus-2, field_modulus)),
	Add=lambda x, y: (x + y) % curve_order,
	Mul=lambda x, y: (x * y) % curve_order,
	ScalarMult=multiply,
	PointNeg=lambda x: (x[0], -x[1]),
	PointInv=lambda x: x.inv(),
	PointAdd=add,
	Hs=hashs,
	Hp=hashp,
	Equal=lambda x, y: x == y,
)


def extract_vars(term):
	return {str(x): rands() for x in T.variables(term)}


def eval_expr(expr, local_vars):
	expr_env = expr_funcs.copy()
	expr_env.update(local_vars)
	return eval(expr, {'__builtins__': None}, expr_env)

# Decompose point operations into underlying arithmetic
DecomposeRules = [
	T.rule("G(x) -> Mul(GDL(), x)"),
	T.rule("G(x) -> Mul(x, GDL())"),

	T.rule("ScalarMult(x, y) -> Mul(x, y)"),
	T.rule("ScalarMult(x, y) -> Mul(y, x)"),

	T.rule("PointAdd(x, y) -> Add(x, y)"),
	T.rule("PointAdd(x, y) -> Add(y, x)"),

	T.rule("PointNeg(x) -> Neg(x)"),
	T.rule("PointInv(x) -> Inv(x)"),

	T.rule("PointDouble(x) -> Add(x,x)"),
]

ExpandRules = [
	T.rule("Mul(Add(a, b), c) -> Add(Mul(a, c), Mul(b, c))"),
	T.rule("Mul(Mul(x, y), z) -> Mul(x, Mul(y, z))"),
]

ReduceRules = [
	T.rule("Add(x, Neg(x)) -> x"),
	T.rule("Mul(Mul(x, y), Inv(y)) -> x"),
	T.rule("Add(Mul(x, y), Mul(x, z)) -> Mul(x, Add(y, z))"),
	T.rule("Add(Mul(x, y), Mul(x, z)) -> Mul(Add(y, z), x)"),
]

RecomposeRules = [
	T.rule("Opaque(x) -> x"),
	T.rule("Mul(GDL(), x) -> TypePoint(G(x))"),
	T.rule("Mul(x, GDL()) -> TypePoint(G(x))"),
	#T.rule("Opaque(G(x)) -> TypePoint(Opaque(G(x)))"),
	T.rule("Mul(TypePoint(x), y) -> TypePoint(ScalarMult(x, y))"),
	T.rule("Mul(y, TypePoint(x)) -> TypePoint(ScalarMult(x, y))"),
	T.rule("Add(TypePoint(x), TypePoint(y)) -> TypePoint(PointAdd(x, y))"),
]

StripTypes = [
	T.rule("TypePoint(x) -> x"),
	T.rule("TypeScalar(x) -> x"),
]
"""
	T.rule("Mul(x, G(y)) -> ScalarMult(G(y), x)"),
	T.rule("Add(G(x), ScalarMult(y, z)) -> PointAdd(G(x), ScalarMult(y, z))"),
	T.rule("Add(ScalarMult(y, z), G(x)) -> PointAdd(ScalarMult(y, z), G(x))"),
	T.rule("Add(ScalarMult(y, z), ScalarMult(x, v)) -> PointAdd(ScalarMult(y, z), ScalarMult(x, v))"),
	T.rule("Add(G(x), G(y)) -> PointAdd(G(x), G(y))"),
	T.rule("Mul(x, PointAdd(y, z)) -> ScalarMult(PointAdd(y, z), x)"),
	T.rule("Mul(G(x), y) -> ScalarMult(G(x), y)"),
	T.rule("Mul(z, ScalarMult(x, y)) -> ScalarMult(ScalarMult(x, y), z)"),
]
"""

def permuteTerms(term):
	dupes = set()
	for u in term.normalforms(DecomposeRules):
		if u not in dupes:
			yield u
			dupes.add(u)
		for v in T.term(u).normalforms(ExpandRules):
			if v not in dupes:
				yield v
				dupes.add(v)
			for w in T.term(v).normalforms(ReduceRules):
				if w not in dupes:
					yield w
					dupes.add(w)
				for r in T.term(w).normalforms(RecomposeRules):
					for s in T.term(r).normalforms(StripTypes):
						if s not in dupes:
							yield s
							dupes.add(s)

if __name__ == "__main__":
	a = T.term("PointAdd(G(w), ScalarMult(G(w), s))")
	a = T.term("ScalarMult(PointAdd(Opaque(G(a)), G(b)), c)")
	#a = T.term("""ScalarMult(ScalarMult(PointAdd(Opaque(G(a)), G(b)), c), z)""")
	#a = T.term("G(Add(Hs(c), Mul(Mul(Add(a, b),c), z)))")
	a = T.withvars(
		T.term("Add(r, s)"), dict(
			s="Add(d, Mul(k, r))",
			r="Add(Mul(k, x), m)",
		))

	print("")
	print("Expr:", a)

	local_vars = extract_vars(a)
	result = eval_expr(str(a), local_vars)

	print("Vars:", ', '.join(local_vars.keys()))
	print("Result:", result)
	print("")
	for u in permuteTerms(a):
		cmp_result = eval_expr(str(u), local_vars)
		if isinstance(cmp_result, long) and isinstance(result, tuple):
			#cmp_result = sbmul(cmp_result)
			continue

		print("=", u)
		if cmp_result != result:
			print("Error: expected:", result)
			print("            got:", cmp_result)
		print("")

	print("")

	#T.show_tree(a, rs)

