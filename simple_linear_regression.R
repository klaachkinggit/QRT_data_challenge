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


#FIRST TRY LINEAR REGRESSION 
# fit linear regression
out <- lm(y ~ ., data = train_df)
print(summary(out))

# diagnostics (compute once)
r <- rstudent(out)
d <- cooks.distance(out)
yhat <- fitted(out)

# sample for fast plotting
set.seed(1)
n_plot <- 50000
idx <- sample.int(length(yhat), size = min(n_plot, length(yhat)))

# plotting settings
par(mfrow = c(2,2))
par(mar = c(4,4,2,1))  # bottom, left, top, right

# 1) Residuals vs fitted
plot(yhat[idx], r[idx], pch = 16, cex = 0.3)
abline(h = 0, col = "red")

# 2) QQ plot (sampled)
qqnorm(r[idx], pch = 16, cex = 0.3)
qqline(r[idx], col = "red")

# 3) Cook's distance (sampled)
plot(d[idx], type = "h")

# 4) Cook's distance vs |residual| (sampled)
plot(d[idx], abs(r[idx]), pch = 16, cex = 0.3)


par(mfrow = c(1,1))
# choose a small set (edit as you like)
vars <- c("y", "RET_1", "RET_2", "RET_3", "SIGNED_VOLUME_2", "SIGNED_VOLUME_3", "SIGNED_VOLUME_4", "GROUP")
# sample rows for speed
set.seed(1)
n_plot <- 5000
idx <- sample.int(nrow(train_df), size = min(n_plot, nrow(train_df)))

pairs(train_df[idx, vars], pch = 16, cex = 0.3)

# --- LOOCV using the hat-matrix trick (PRESS) ---

# y true values
y_true <- train_df$y

# In OLS: yhat = H y, residuals e = (I - H) y
e <- resid(out)              # e = y - yhat
h <- hatvalues(out)          # leverage = diag(H), computed efficiently from QR (no full H)

# Avoid division by zero if leverage is extremely close to 1
den <- pmax(1 - h, 1e-12)

# LOOCV prediction for each point i:
# yhat_loo_i = y_i - e_i / (1 - h_ii)
yhat_loo <- y_true - (e / den)

# --- Convert to sign prediction ---
# Choose one convention:
# 1) {0,1} labels:
y_sign      <- as.integer(y_true > 0)
yhat_sign   <- as.integer(yhat_loo > 0)

# 2) Or {-1,+1} labels (uncomment if you prefer)
# y_sign    <- ifelse(y_true > 0,  1, -1)
# yhat_sign <- ifelse(yhat_loo > 0, 1, -1)

# --- Accuracy ---
acc <- mean(yhat_sign == y_sign)
cat("LOO sign accuracy (hat-matrix trick):", acc, "\n")

# ---------- Predict on X_test as 0/1 and save ----------

# Continuous prediction (can be any real number)
pred_test_cont <- as.numeric(predict(out, newdata = X_test))
sum(is.na(pred_test_cont))

# Convert to 0/1 based on sign
pred_test_01 <- as.integer(pred_test_cont > 0)

linear_regression_pred <- data.frame(
  ROW_ID = rownames(X_test),
  PREDICTION = pred_test_01,
  check.names = FALSE
)

data_dir <- normalizePath("~/Desktop/QRT/submission", mustWork = TRUE)
out_path <- file.path(data_dir, "linear_regression_pred.csv")

write.csv(linear_regression_pred, out_path, row.names = FALSE)
cat("Saved 0/1 predictions to:", out_path, "\n")

