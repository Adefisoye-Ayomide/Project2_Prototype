

class LongFloat():
	
	def __init__(self, sig_digits, exp):
		self.sig_digits = sig_digits
		self.exp = exp

	def __mul__(self, other):
		sig = self.sig_digits * other.sig_digits
		exp = self.exp * other.exp

	def print(self, string):
		print(self.sig_digits * 10 ** self.exp)
		

