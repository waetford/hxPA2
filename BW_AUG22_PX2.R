#Ben Waetford
#August 2022 Harvard X assignment two

###############################################################################
#load libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(readxl)) install.packages("readxl", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot)) install.packages("rpart.plot", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(naivebayes)) install.packages("naivebayes", repos = "http://cran.us.r-project.org")
if(!require(partykit)) install.packages("partykit", repos = "http://cran.us.r-project.org")

###############################################################################
#read and format data
my_data <- read_excel("data.xlsx")
head(my_data)

#change column headers and make categorical data for analysis purposes (since I want to use classification trees)
dat <- my_data %>%
  mutate(result = as.factor(Result),
         ext_mgmt = as.factor(External_mgmt),
         client_b = as.factor(Client_base),
         contract_t = as.factor(Client_base),
         contractor_t = as.factor(Contractor_type),
         delivery_m = as.factor(Delivery_model),
         state = as.factor(State),
         sector = as.factor(Sector),
         recur_r = as.factor(Recur_revenue),
         service_f = as.factor(Service_focus),
         syear = as.factor(start_year),
         order_m = as.factor(OM),
         Result = NULL,
         External_mgmt= NULL,
         Client_base = NULL,
         Contract_type = NULL,
         Contractor_type = NULL,
         Delivery_model= NULL,
         State = NULL,
         Sector = NULL,
         Recur_revenue = NULL,
         Service_focus = NULL,
         start_year = NULL,
         OM = NULL)

#check for na
na.s <- sapply(dat, {function(x) any(is.na(x))}) 
knitr::kable(na.s)

#How many rows of data?
nrow(dat)

#how many jobs fall into each result category
count_by_result <- dat %>% 
  group_by(result) %>% 
  summarise(n = n()) %>%
  plot()

#will re-order result factor, so that levels are ordered from worst to best and the plot is more intuitive to interpret
dat$result <- factor(dat$result, levels = c('Big Loss', 'Loss', 'Close Enough', 'Win', 'Big Win'))

dat %>% 
  group_by(result) %>% 
  summarise(n = n()) %>%
  ggplot(aes(result, n)) +
  geom_col()


#create column that summarizes the results of each job (this new variable will be the dependent variable in the machine learning analysis for this individual assignment) 
dat <- dat %>%
  mutate(category = as.factor(ifelse(dat$result %in% c('Close Enough', 'Win', 'Big Win'), "Better than expected", "Worse than expected"))
  )

#look at the new variable

ggplot(dat, aes(category)) +
  geom_bar() +
  ggtitle("Count of jobs by category")
  
ggplot(dat, aes(order_m)) + 
  geom_bar(position = "fill", aes(fill = category))


###############################################################################
# generate training and test sets
set.seed(1976, sample.kind = 'Rounding') #make repeatable
ind <- sample(2, nrow(dat), replace = T, prob = c(0.7, 0.3)) #generate indexes for each set
train <- dat[ind==1,] #create training set
test <- dat[ind==2,] #create test set

###############################################################################
#create a table that will be used to record the accuracy of each model, the rate to beat is the no information rate
accuracy_table <- data_frame(Model = "No information rate", Rate = 0.7199)

train <- train %>% select(-c(result))
test <- test %>% select(-c(result))

###############################################################################
#ANALYSIS
###############################################################################

#randomForest's randomForest function
set.seed(1979, sample.kind = 'Rounding') #make results repeatable
rf <- randomForest(category ~ ., data = train) #fit model
p.rf <-predict(rf, test) #prediction
rF.a <- confusionMatrix(p.rf, test$category)$overall["Accuracy"] #compute accuracy
accuracy_table <- bind_rows(accuracy_table, data_frame(Model="randomForest", Rate  = rF.a)) #add score to accuracy table


#caret's train function
set.seed(1979, sample.kind = 'Rounding') #make results repeatable
caret_train <- train(category ~., data = train, method = "rf") #fit model
p.caret <-predict(caret_train, test) #prediction
cm.caretrf <- confusionMatrix(p.caret, test$category)$overall["Accuracy"] #compute accuracy
accuracy_table <- bind_rows(accuracy_table, data_frame(Model="caret rf", Rate  = cm.caretrf)) #add score to accuracy table

####rpart random forest
set.seed(1976, sample.kind = 'Rounding') #make results repeatable
rf_rpart <- rpart(category ~ ., train) #fit model
p.rpart <- predict(rf_rpart, test, type = 'class') #prediction
cm.rpart <- confusionMatrix(p.caret, test$category)$overall["Accuracy"] #compute accuracy
accuracy_table <- bind_rows(accuracy_table, data_frame(Model="rpart", Rate  = cm.rpart)) #add score to accuracy table

####ctree
set.seed(1976, sample.kind = 'Rounding') #make results repeatable
t.ctree <- ctree(category ~., data = train) #fit model
p.ctree <-predict(t.ctree, test, type = 'response') #prediction
cm.ctree <- confusionMatrix(p.ctree, test$category)$overall["Accuracy"] #compute accuracy
accuracy_table  <- bind_rows(accuracy_table, data_frame(Model="ctree", Rate  = cm.ctree)) #add score to accuracy table

#knn
set.seed(1976, sample.kind = 'Rounding') #make results repeatable
c.knn <- train_knn <- train(category ~ ., method = "knn", train) #fit model
p.knn <- predict(c.knn, test) #prediction
cm.knn <- confusionMatrix(p.knn, test$category)$overall["Accuracy"] #compute accuracy
accuracy_table  <- bind_rows(accuracy_table, data_frame(Model="knn", Rate  = cm.knn)) #add score to accuracy table

##Naive bayes
set.seed(1976, sample.kind = 'Rounding') #make results repeatable
nb <- naive_bayes(category ~ ., data = train, laplace = 1) #fit model
p.nb <- predict(nb, test, type = 'class') #prediction
cm.nb <- confusionMatrix(p.nb, test$category)$overall["Accuracy"] #compute accuracy


accuracy_table  <- bind_rows(accuracy_table,data_frame(Model="Naive Bayes",Rate  = cm.nb)) #add score to accuracy table

#accuracy table summary
summary(accuracy_table)

#most accurate models sensitivity and specificity
#randomForest
confusionMatrix(p.rf, test$category)$byClass[1:2]
#knn
confusionMatrix(p.knn, test$category)$byClass[1:2]
