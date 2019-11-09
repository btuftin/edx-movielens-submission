# ! Warning. The analysis takes a long time and requires a lot of memory.
# I recommend restarting R before running it.
# And if you are low on ram it might still give errors when trying to allocate
# vectors several gigabytes large. It does work though. ;)

###################################
# Loading packages and data       #
###################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(recommenderlab)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# Downloading and creating the training and validation sets takes a while, so we will store
# the objects to disk and read them from there if available
f <- "movielens.RData"
if (file.exists(f)) {
  load(f)
} else {
  # This code to create the training and validation sets was supplied
  # to all students
  
  ################################
  # Create edx set, validation set
  ################################
  
  # Note: this process could take a couple of minutes
  
  
  # MovieLens 10M dataset:
  # https://grouplens.org/datasets/movielens/10m/
  # http://files.grouplens.org/datasets/movielens/ml-10m.zip
  
  dl <- tempfile()
  download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
  
  ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                   col.names = c("userId", "movieId", "rating", "timestamp"))
  
  movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
  colnames(movies) <- c("movieId", "title", "genres")
  movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                             title = as.character(title),
                                             genres = as.character(genres))
  
  movielens <- left_join(ratings, movies, by = "movieId")
  
  # Validation set will be 10% of MovieLens data
  # If executing on an older version of R use:
  # set.seed(1)
  set.seed(1, sample.kind="Rounding")
  
  test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
  edx <- movielens[-test_index,]
  temp <- movielens[test_index,]
  
  # Make sure userId and movieId in validation set are also in edx set
  
  validation <- temp %>% 
    semi_join(edx, by = "movieId") %>%
    semi_join(edx, by = "userId")
  
  # Add rows removed from validation set back into edx set
  
  removed <- anti_join(temp, validation)
  edx <- rbind(edx, removed)
  
  rm(dl, ratings, movies, test_index, temp, movielens, removed)
  
  save(edx, validation, file = f)
}

#############################################
# Building basic model                      #
#############################################

edx_mean <- mean(edx$rating)

l <- 5 # regularization factor chosen based on training

movie_effect <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - edx_mean)/(n()+l))

user_effect <- edx %>% 
  left_join(movie_effect, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - edx_mean)/(n()+l))

#############################################
# Partial SVD analysis                      #
#############################################

#############################################
# Function partialFunkSVD() based on the    #
# funkSVD() function in the recommenderlab  #
# package
#############################################

partialFunkSVD <- function(x, k = 10, U = NULL, V=NULL, gamma = 0.015, 
                           lambda = 0.001, min_improvement = 1e-6, 
                           min_epochs = 50, max_epochs = 200, 
                           verbose = FALSE) {
  
  x <- as(x, "matrix")
  
  if (ncol(x) < k || nrow(x) < k)
    stop("k needs to be smaller than the number of users or items.")
  
  # initialize the user-feature and item-feature matrix if they're not parameters
  if(is.null(U)) {
    U <- matrix(0.1, nrow = nrow(x), ncol = k)
    updtU = TRUE
  } else {
    updtU = FALSE
    U <- as.matrix(U)[1:nrow(x),]
  }
  
  if(is.null(V)) {
    V <- matrix(0.1, nrow = ncol(x), ncol = k)
    updtV = TRUE
  } else {
    updtV = FALSE
    V <- as.matrix(V)[1:ncol(x),]
  }
  
  #list of indices pointing to ratings on each item
  itemIDX <- lapply(1:nrow(x), function(temp) which(!is.na(x[temp, ])))
  #list of indices pointing to ratings on each user
  userIDX <- lapply(1:ncol(x), function(temp) which(!is.na(x[, temp])))
  
  # go through all features
  for (f in 1:k) {
    if(verbose) cat("\nTraining feature:", f, "/", k, ": ")
    
    # convergence check
    last_error <- Inf
    delta_error <- Inf
    epoch <- 0L
    p <- tcrossprod(U, V)
    
    while (epoch < min_epochs || (epoch < max_epochs &&
                                  delta_error > min_improvement)) {
      
      # update user features
      error <- x - p
      temp_U <- as.matrix(U)
      
      if(updtU) {
        for (j in 1:ncol(x)) {
          delta_Uik <- lambda * (error[userIDX[[j]], j] * V[j, f] -
                                   gamma * U[userIDX[[j]], f])
          U[userIDX[[j]], f] <- U[userIDX[[j]], f] + delta_Uik
        }
      }
      
      # update item features
      if(updtV) {
        for (i in 1:nrow(x)) {
          delta_Vjk <- lambda * (error[i, itemIDX[[i]]] * temp_U[i, f] -
                                   gamma * V[itemIDX[[i]], f])
          V[itemIDX[[i]], f] <- V[itemIDX[[i]], f] + delta_Vjk
        }
      }
      
      ### update error
      p <- tcrossprod(U, V)
      new_error <- sqrt(sum(abs(x - p)^2, na.rm = TRUE)/length(x))
      delta_error <- abs(last_error - new_error)
      
      last_error <- new_error
      epoch <- epoch + 1L
      if(verbose) cat(".")
      
      gc() # Manually calling garbage collection because of memory issues
    }
    
    if(verbose) cat("\n-> ", epoch, "epochs - final improvement was",
                    delta_error, "\n")
  }
  
  structure(list(U = U, V = V, parameters =
                   list(k = k, gamma = gamma, lambda = lambda,
                        min_epochs = min_epochs, max_epochs = max_epochs,
                        min_improvement = min_improvement)),
            class = "funkSVD")
}

##############################################
# Creating matrix for partial SVD analysis   #
##############################################

# This takes a while because of the size of the matrix
# so the script saves the variables to disk at the end
# and reads them from disk if it's run again
f <- "movielens_matrix.RData"
if(exists(f)) {
  load(f)
} else {
  # turn the ratings data_frame into a matrix of userId x movieId
  m <- edx %>% pivot_wider(c("userId", "movieId"), names_from = movieId, values_from = rating, values_fill = list(rating = NA)) %>% as.matrix()
  
  
  # We no longer need the edx object, and we need as much memory space as
  # we can get
  rm(edx)
  gc()
  
  # The first column has our userIds, we'll make those rownames and remove the column
  rownames(m) <- m[,1]
  m <- m[,-1]
  
  # Sweep out the overall, row and column means
  m_mean <- mean(m, na.rm = TRUE)
  m_res <- m - m_mean
  m_col_means <- colMeans(m_res, na.rm=TRUE)
  m_res <- sweep(m_res, 2, colMeans(m_res, na.rm=TRUE))
  m_row_means <- rowMeans(m_res, na.rm=TRUE)
  m_res <- sweep(m_res, 1, rowMeans(m_res, na.rm=TRUE))
  
  # We take the matrix of residuals and sort it by density in both directions
  m_res <- m_res[order(apply(!is.na(m_res), 1, sum), decreasing = TRUE),
                        order(apply(!is.na(m_res), 2, sum), decreasing = TRUE)]
  
  # deleting large unused objects
  rm(m, m_col_means, m_row_means)
  gc() # Manually calling garbage collection because of memory issues

  save(m_res, file = f)
  gc() # Manually calling garbage collection because of memory issues
}

#######################################################
# Run partialFunkSVD-analysis on subset of matrix     #
#######################################################


# analysis variables for full analysis
 k <- 5
 max_epochs <- 200
 min_epochs <- 50
 verbose <- TRUE
 proportion <- 30 # (in %) Proportion of data along each dimension to use in analysis


# Find the row and column number for our cutoff point
dimm <- (ncol(m_res) * proportion) %/% 100
dimu <- (nrow(m_res) * proportion) %/% 100

# We run partialFunkSVD for all movies, but with only the most prolific
# users
# This takes a while to run, to preserve state in case of unexpected computer
# failure it's saved to disk and read from there if rerunning analysis
f <- "movielens-analysis-results1.RData"
if (exists(f)) {
  load(f)
} else {
  decomp <- partialFunkSVD(m_res[1:dimu,], k = k, min_epochs = min_epochs, max_epochs = max_epochs, verbose = verbose)
  gc() # Manually calling garbage collection because of memory issues
  save(decomp, file = f)
}

gc() # Manually calling garbage collection because of memory issues


# We run partialFunkSVD for all users, but supplying the movie
# patterns established in the previous call, and create user
# patterns based on the patterns for the most rated movies
# This takes a while to run, to preserve state in case of unexpected computer
# failure it's saved to disk and read from there if rerunning analysis

f <- "movielens-analysis-results2.RData"
if (exists(f)) {
  load(f)
} else {
  decomp2 <- partialFunkSVD(m_res[, 1:dimm], V = decomp$V[1:dimm,], k = k, min_epochs = min_epochs, max_epochs = max_epochs, verbose = verbose)#)
  gc() # Manually calling garbage collection because of memory issues
  save(decomp, file = f)
}


#################################################
# Calculating RMSE for the test set             #
#################################################

# We need column and row names added to our decomposition to get
# values for specific movies and users
rownames(decomp$V) <- colnames(m_res)
rownames(decomp2$U) <- rownames(m_res)

# Calculate square error for each rating in validation set and sum them
SE <- 0
for(i in 1:nrow(validation)){
  user <- validation[i,1]
  movie <- validation[i,2]
  rating <- validation[i,3]
  
  # Add the overall mean, movie effect and user effect to create prediction
  pred <- m_mean + 
    movie_effect %>% filter(movieId == movie) %>% pull(b_i) +
    user_effect %>% filter(userId == user) %>% pull(b_u)
  
  # Add the residual predicted by SVD for this user and movie
  
  pred <- pred + as.numeric(decomp2$U[(as.character(user)),] %*% decomp$V[(as.character(movie)),])
  
  # Add term to the total square error
  SE <- SE + (rating - pred)^2
}

# Calculate RMSE
rmse <- sqrt(SE/nrow(validation))

# Output result
cat("\n", "The rmse for this analysis is: ", rmse, "\n")
# The rmse for this analysis is:  0.8115521 
