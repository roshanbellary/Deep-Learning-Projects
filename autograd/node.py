from __future__ import annotations

class Tensor():
    def __init__(self, value : int, _children = (), requires_grad = True):
        self.value = value
        self._prev = set(_children)
        self._backward = lambda : None
        self.requires_grad = requires_grad 
        self.grad = 0.0

    def __add__(self, other : Tensor):
        other = Tensor(other, requires_grad=False) if not isinstance(other, Tensor) else other
        
        t = Tensor(value = self.value + other.value, _children = (self, other)) 
        def _backward():
            self.grad += 1.0 * t.grad
            other.grad += 1.0 * t.grad
        t._backward = _backward
        return t

    def __mul__(self, other: Tensor):
        other = Tensor(other, requires_grad=False) if not isinstance(other, Tensor) else other
        t = Tensor(value = self.value * other.value, _children = (self, other))    
        def _backward():
            self.grad += other.value * t.grad
            other.grad += self.value * t.grad

        t._backward = _backward
        return t
    def __rmul__(self, other: Tensor):
        return self.__mul__(other)

    def __sub__(self, other: Tensor):
        other = Tensor(other) if not isinstance(other, Tensor) else other
        t = Tensor(value = self.value - other.value, _children = (self, other))
        def _backward():
            self.grad += t.grad
            other.grad -= t.grad
        t._backward = _backward
        return t

    def backward(self):
        self.grad = 1
        topo : list[Tensor]= []
        visited = set()
        def build_topo(n):
            for c in n._prev:
                if c not in visited and c.requires_grad:
                    build_topo(c)
            topo.append(n)
            visited.add(n)
        build_topo(self)
        for n in reversed(topo):
            n._backward()
    def __str__(self) -> str:
        return f"Tensor(value={self.value}, requires_grad={self.requires_grad}, grad={self.grad})" 
if __name__ == "__main__":
   a = Tensor(2)
   v = a * 2
   z = 2 * a
   print(v.value)
   print(z.value)
   z.backward()
   print(a.grad)
   for child in z._prev:
       print(child)