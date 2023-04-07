import multiply
import divide
import add

x = int(input())
y = int(input())

print(x,"+",y,"=",add(x,y))
print(x,"-",y,"=",add(x,-y))
print(x,"x",y,"=",multiply(x,y))
if y!=0 :
  print(x,"/",y,"=",divide(x,y))
else:
  print("you cannot divide by 0")
