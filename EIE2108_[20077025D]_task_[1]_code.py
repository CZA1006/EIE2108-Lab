class Quaternion:
    """A class defines quaternion number"""

    def __init__(self,value):
        if type(value)!=list or len(value)!=4:
            print('Err: It must be in a list of 4 values!')
        else:
            self.value = value

    """Realize the addition of two quaternions"""
    def add(self,other):
        return Quaternion([self.value[0]+other.value[0],self.value[1]+other.value[1],self.value[2]+other.value[2],self.value[3]+other.value[3]])

    """Realize the multiplication of two quaternions"""
    def mul(self,other):
        a = self.value[0] * other.value[0] - self.value[1] * other.value[1] - self.value[2] * other.value[2] - self.value[3] * other.value[3]
        b = self.value[0] * other.value[1] + self.value[1] * other.value[0] + self.value[2] * other.value[3] - self.value[3] * other.value[2]
        c = self.value[0] * other.value[2] - self.value[1] * other.value[3] + self.value[2] * other.value[0] + self.value[3] * other.value[1]
        d = self.value[0] * other.value[3] + self.value[1] * other.value[2] - self.value[2] * other.value[1] + self.value[3] * other.value[0]
        return Quaternion([a,b,c,d])

    """Get the conjugate as a quaternion number"""
    def conj(self):
        return Quaternion([self.value[0],-self.value[1],-self.value[2],-self.value[3]])

    # To avoid Quaternion by Float, I split the numerator into four parts and divide the denominator.
    """Get the inverse of a quaternions"""
    def inv(self):
        return Quaternion([(self.value[0])/((self.value[0])**2 + (self.value[1])**2 + (self.value[2])**2 + (self.value[3])**2),(-self.value[1])/((self.value[0])**2 + (self.value[1])**2 + (self.value[2])**2 + (self.value[3])**2),(-self.value[2])/((self.value[0])**2 + (self.value[1])**2 + (self.value[2])**2 + (self.value[3])**2),(-self.value[3])/((self.value[0])**2 + (self.value[1])**2 + (self.value[2])**2 + (self.value[3])**2)])

    """Get the norm of a quaternion"""
    def norm(self):
        tmp=0
        for item in self.value:
            tmp += item**2
        return (tmp**0.5) # Take the square root of the sum of terms

# create z1, z2, z3
z1 = Quaternion([1,2,3,4])
z2 = Quaternion([2,3,4,5])
z3 = Quaternion([3,4,5,6])
# create z4 to get -z3
z4 = Quaternion([-1,0,0,0])
print(z3.mul(z2).value) # z3z2
z5 = z4.mul(z3)
print(z1.add(z5).value) # z1-z3
z6 = z1.conj()
print(z6.mul(z3).value) # z1*z3
z7 = z3.inv()
print(z7.mul(z7).mul(z7).value) # z3^-3
print(z2.norm()) #||z2||
"""the final answer is complicated, we divided it into some parts."""
z8 = z1.mul(z1).mul(z1)
z9 = z2.conj()
z10 = z7.mul(z7).mul(z7).mul(z7)
z11 = z1.add(z3)
z12 = z11.mul(z11)
z13 = z8.mul(z9).mul(z10).mul(z12)
z14 = z1.add(z2)
z15 = z2.norm()
z16 = Quaternion([z15,0,0,0])
z17 = z14.mul(z16)
z18 = z17.inv()
z19 = z13.mul(z18)
print(z19.value) # final answer









