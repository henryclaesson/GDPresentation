from sympy import *
 
def fun():
    x_1, x_2 = symbols('x_1 x_2')      
    syms = [x_1, x_2]                      
    f = (6.983 * (x_1**2)) + (12.415 * (x_2 **2)) - x_1        
    return f, syms 

def objF(f, syms, x_K):
    variableArray = syms                                                                

    c = Matrix([f]).jacobian(variableArray)                                            
    gradVal = c.subs([((variableArray[i], x_K[i])) for i in range(len(x_K))])          
    return  gradVal

def steepestDescent(f, syms, k, x_K, aK, convergenceEpsion):
    while 1:
        cK = objF(f, syms, x_K)           
        cnorm = (objF(f, syms, x_K)).norm(2)                                  

        if cnorm > convergenceEpsion:                       
            dk = -cK                                        
            for n in range(len(x_K)):
                x_K[n] = x_K[n] + (aK * dk[n])              
            k = k + 1                                       
        else:
            break
        
    print("The solution using steepest descent is: " + str(x_K) + "\n It took " + str(k) + " itterations to get to the solution\n\n")

def conjugateGradient(f, syms, k, x_K, aK, convergenceEpsion):  
    while 1:
        cK = objF(f, syms, x_K)
        cnorm = (cK).norm(2)
        normcKLast = cnorm

        if cnorm > convergenceEpsion:
            dK = -cK
            dKLast = dK
            dK = -cK + ((cnorm/normcKLast)**2)*dKLast
            
            for n in range(len(x_K)):
                x_K[n] = x_K[n] + (aK*dK[n])
            k = k + 1
        
        else:
            break
    print("The solution using conjugate gradient is: " + str(x_K) + "\n It took " + str(k) + " itterations to get to the solution\n\n")

def modifiedNewton(f, syms, k, x_K, aK, convergenceEpsion):
    x_1 = syms[0]
    x_2 = syms[1]

    while 1:
        cK = objF(f, syms, x_K)
        cnorm = cK.norm(2)
        if cnorm > convergenceEpsion:
            dK = -1 * ((hessian(f, (x_1, x_2))).inv()) * cK.T
            for n in range(len(x_K)):
                x_K[n] = x_K[n] + (aK*dK[n])
            k = k + 1
        else:
            break
    print("The solution using modified newton is: " + str(x_K) + "\n It took " + str(k) + " itterations to get to the solution\n\n")


def DFP(f, syms, k, x_K, aK, convergenceEpsion):
    A = eye(2,2)                          

    while 1:
        cK = objF(f, syms, x_K)       
        cnorm = cK.norm(2)

        if cnorm > convergenceEpsion:       
            dK = -1 * A * cK.T      
            previousgradval = cK    
            deltaXk = Matrix([(aK * dK[n]) for n in range(len(x_K))])      
            x_K = [(x_K[n] + deltaXk[n]) for n in range(len(x_K))]      
            cK = objF(f, syms, x_K)
            A = A + ((deltaXk * deltaXk.T) / ((deltaXk.T * (cK - previousgradval).T).norm(2))) + (-1 * (((A * (cK - previousgradval).T) * 
                        (A * (cK - previousgradval).T).T)/(((cK - previousgradval) * (A * (cK - previousgradval).T)).norm(2))))
            k = k + 1

        else:
            break
    print("The solution using DFP is: " + str(x_K) + "\n It took " + str(k) + " itterations to get to the solution\n\n")


x_K = [2, 1]                # initial guess
convergenceEpsion = 0.01    # tolerance for convergence
k = 0                       # initialize itteration counter
aK = 0.001                  # LEARNING RATE: CHANGE THIS TO SEE HOW IT AFFECTS HOW QUICKLY YOU CONVERGE
f, syms = fun()             # initial caluclation for working parameters for function

# steepestDescent(f, syms, k, x_K, aK, convergenceEpsion)

# conjugateGradient(f, syms, k, x_K, aK, convergenceEpsion)

# modifiedNewton(f, syms, k, x_K, aK, convergenceEpsion)

