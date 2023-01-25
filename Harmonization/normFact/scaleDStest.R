scaleDStest  <- function(b,x,means,stds){
  
  # center and scale each feature by batch
  
  # x[i,j] gives value of feature j for sample i
  # b(i) gives batch number of sample i
  # status = TRUE, training process; status = FALSE, test process
  
  
    ds = unique(b)
    xc = x*0-1
    for (i in ds){
      ix = which(b==i)
      x_batch = x[ix,]
      xc[ix,] = (x_batch-means[i,])/stds[i,]
    }

    return(xc)

}