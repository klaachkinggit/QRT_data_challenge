#Additive models TAKES TOO MUCH TIME TO RUN SWITCHED ON PYTHON
# ---------- Robust data prep (ROW_ID-safe) ----------

# Read with ROW_ID as rownames (prevents silent misalignment)
X_train <- read.csv("~/Desktop/QRT/data/X_train.csv", row.names = "ROW_ID", check.names = FALSE)
X_test  <- read.csv("~/Desktop/QRT/data/X_test.csv",  row.names = "ROW_ID", check.names = FALSE)
y_train <- read.csv("~/Desktop/QRT/data/y_train.csv", row.names = "ROW_ID", check.names = FALSE)

# Align training rows by ROW_ID (critical!)
common_train_ids <- intersect(rownames(X_train), rownames(y_train))
X_train <- X_train[common_train_ids, , drop = FALSE]
y_train <- y_train[common_train_ids, , drop = FALSE]

# (Optional sanity check)
stopifnot(identical(rownames(X_train), rownames(y_train)))

# Drop columns (only those that exist)
cols_to_drop <- c("SIGNED_VOLUME_1", "SIGNED_VOLUME_20", "MEDIAN_DAILY_TURNOVER", "ALLOCATION", "TS")
drop_train <- intersect(cols_to_drop, colnames(X_train))
drop_test  <- intersect(cols_to_drop, colnames(X_test))

X_train <- X_train[, !(colnames(X_train) %in% drop_train), drop = FALSE]
X_test  <- X_test[,  !(colnames(X_test)  %in% drop_test),  drop = FALSE]

# Ensure GROUP is a factor (and keep same levels in test)
if ("GROUP" %in% colnames(X_train)) {
  X_train$GROUP <- factor(X_train$GROUP)
}
if ("GROUP" %in% colnames(X_test) && "GROUP" %in% colnames(X_train)) {
  X_test$GROUP <- factor(X_test$GROUP, levels = levels(X_train$GROUP))
}

# Impute NA using TRAIN means (numeric columns only)
num_cols <- sapply(X_train, is.numeric)
means <- colMeans(X_train[, num_cols, drop = FALSE], na.rm = TRUE)

for (nm in names(means)) {
  X_train[[nm]][is.na(X_train[[nm]])] <- means[[nm]]
  X_test[[nm]][is.na(X_test[[nm]])]   <- means[[nm]]
}

# Build training data frame (y + X)
y <- y_train[[1]]
train_df <- data.frame(y = y, X_train, check.names = FALSE)

# Done: train_df is robustly aligned & ready
str(train_df)

# Convert the continuous 'y' into a binary factor (1 for positive, 0 for negative/zero)
train_df$y_binary <- as.factor(ifelse(train_df$y > 0, 1, 0))

# Now check the balance of our new categories!
table(train_df$y_binary)
prop.table(table(train_df$y_binary))


library(randomForest)

# use y_binary as our target. 
# subtract the original continuous 'y' so the model doesn't cheat!
rf_model = randomForest(y_binary ~ . - y, data = train_df, ntree = 100, importance = TRUE)

# View the out-of-bag error and model summary
print(rf_model)

