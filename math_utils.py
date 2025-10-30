def find_max_number(n1, n2, n3):
    if n1 > n2 :
      if n1 > n3 :
        return n1 
      else :
        return n3
    else :
      if n2 > n3 :
        return n2
      else :
        return n3

def find_mean(num1, num2, num3):
    mean = (num1 + num2 + num3)/3
    return mean

def find_mean_std(num1, num2, num3):
    mean = find_mean(num1, num2, num3)
    std = (((num1 - mean)**2 + (num2 -mean)**2 +(num3 - mean)**2)/3)**0.5
    return mean, std
