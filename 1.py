class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out

# Test
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)

e = a * b
d = e + c

print("d.data:", d.data)
print("d._prev:", d._prev)
print("d._op:", d._op)

# ----------------------------------------------------------------------------------------------------------------------------------
# Graphviz ile Görselleştirme (opsiyonel)
from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        # her Value için bir nokta (node) oluştur
        dot.node(name=uid, label=f"{{ data {n.data:.4f} }}", shape='record')
        if n._op:
            # işlem varsa, ayrı bir "operasyon düğümü" oluştur
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # n1'den n2'nin operasyon düğümüne bağlan
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

# Kullanım:
dot = draw_dot(d)
dot.render('expression_graph', view=False)  # expression_graph.svg dosyası oluşturur