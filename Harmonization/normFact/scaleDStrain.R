scaleDStrain  <- function(b,x){
  
  # center and scale each feature by batch
  
  # x[i,j] gives value of feature j for sample i
  # b(i) gives batch number of sample i
 
    
    ds = unique(b)
    xc = x*0-1
    means = matrix(data=0,nrow(ds),ncol(x))
    stds = means
    for (i in ds){
      ix = which(b==i)
      x_batch = x[ix,]
      xc[ix,] = scale(x_batch)
      x_batch = scale(x_batch)
      means[i,] = attr(x_batch,"scaled:center")
      stds[i,] = attr(x_batch,"scaled:scale") # root mean square values.
    }
    
    out = list(xc=xc,means=means,stds=stds)
    return(out)
  
}