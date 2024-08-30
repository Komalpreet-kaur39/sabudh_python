def replace_last_tuple(tup,new_value):
    tuple2=[]
    for t in tup:
        new_tuple=t[:-1]+(new_value,)   
        tuple2.append(new_tuple)
    return tuple2
tuple1=[(12,3,4,2),(4,2,3,4),(3,4,4,2)]
print("\n",tuple1)
result=replace_last_tuple(tuple1,100)
print("After replacing the last value of tuples in a list",result)