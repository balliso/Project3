library(plumber)
library(tidyverse)
library(tidymodels)

# Read in data
diabetes_df <- read_csv("../diabetes_012_health_indicators_BRFSS2015.csv", show_col_types = FALSE)

# Convert categorical variables into factors
diabetes_df <- diabetes_df |>
  mutate(
    Diabetes_012 = factor(Diabetes_012, levels = c(0,1,2), labels = c("No Diabetes", "Prediabetes", "Diabetes")),
    HighBP = factor(HighBP, labels = c("No High BP", "High BP")),
    HighChol = factor(HighChol, labels = c("No High Chol", "High Chol")),
    CholCheck = factor(CholCheck, labels = c("No Check", "Checked")),
    Smoker = factor(Smoker, labels = c("Non-Smoker", "Smoker")),
    Stroke = factor(Stroke, labels = c("No Stroke", "Stroke")),
    HeartDiseaseorAttack = factor(HeartDiseaseorAttack, labels = c("No HD/Attack", "HD/Attack")),
    PhysActivity = factor(PhysActivity, levels = c(0, 1), labels = c("No Activity", "Activity")),
    Fruits = factor(Fruits, levels = c(0, 1), labels = c("No", "Yes")),
    Veggies = factor(Veggies, levels = c(0, 1), labels = c("No", "Yes")),
    HvyAlcoholConsump = factor(HvyAlcoholConsump, levels = c(0, 1), labels = c("No", "Yes")),
    AnyHealthcare = factor(AnyHealthcare, levels = c(0, 1), labels = c("No", "Yes")),
    NoDocbcCost = factor(NoDocbcCost, levels = c(0, 1), labels = c("No", "Yes")),
    GenHlth = factor(GenHlth, labels = c("Excellent","Very good","Good","Fair","Poor")),
    DiffWalk = factor(DiffWalk, levels = c(0, 1), labels = c("No Difficulty", "Difficulty Walking")),
    Sex = factor(Sex, labels = c("Female", "Male")),
    Age = factor(Age, levels = 1:13, labels = c("18–24", "25–29", "30–34", "35–39", "40–44", "45–49", "50–54", "55–59", "60–64", "65–69", "70–74", "75–79", "80+")),
    Education = factor(Education, levels = 1:6, labels = c("Kindergarten or less", "Grades 1–8", "Grades 9–11", "High school or GED", "Some college/technical school", "College graduate")),
    Income = factor(Income, levels = 1:8, labels = c("<$10,000", "$10,000–<$15,000", "$15,000–<$20,000", "$20,000–<$25,000", "$25,000–<$35,000", "$35,000–<$50,000", "$50,000–<$75,000", "$75,000+"))
  )

# Define best model and recipe
rf_spec <- rand_forest(
  mode = "classification",
  mtry = 2,        # best mtry from tuning
  trees = 100      # same as in Modeling.qmd
) |>
  set_engine("ranger")

rf_recipe <- recipe(
  Diabetes_012 ~ BMI + Age + GenHlth + HighBP + HighChol +
    PhysActivity + DiffWalk + Sex + Education + Income,
  data = diabetes_df
)

rf_wf <- workflow() |>
  add_model(rf_spec) |>
  add_recipe(rf_recipe)

# Fit final model on the entire dataset
final_model <- fit(rf_wf, data = diabetes_df)

# Save factor levels for later
Age_levels <- levels(diabetes_df$Age) 
GenHlth_levels <- levels(diabetes_df$GenHlth) 
HighBP_levels <- levels(diabetes_df$HighBP) 
HighChol_levels <- levels(diabetes_df$HighChol) 
PhysActivity_levels <- levels(diabetes_df$PhysActivity) 
DiffWalk_levels <- levels(diabetes_df$DiffWalk) 
Sex_levels <- levels(diabetes_df$Sex) 
Education_levels <- levels(diabetes_df$Education) 
Income_levels <- levels(diabetes_df$Income)

# Helper mode function for categorical variables
Mode <- function(x) {
  tab <- table(x)
  names(tab)[which.max(tab)]
}

#* @apiTitle Diabetes Prediction API
#* @apiDescription This API...

#* Predict diabetes status using the final random forest model
#*
#* @param BMI
#* @param Age
#* @param GenHlth
#* @param HighBP
#* @param HighChol
#* @param PhysActivity
#* @param DiffWalk
#* @param Sex
#* @param Education
#* @param Income
#* @get /pred
function(
    BMI = mean(diabetes_df$BMI),
    Age = Mode(diabetes_df$Age),
    GenHlth = Mode(diabetes_df$GenHlth),
    HighBP = Mode(diabetes_df$HighBP),
    HighChol = Mode(diabetes_df$HighChol),
    PhysActivity = Mode(diabetes_df$PhysActivity),
    DiffWalk = Mode(diabetes_df$DiffWalk),
    Sex = Mode(diabetes_df$Sex),
    Education = Mode(diabetes_df$Education),
    Income = Mode(diabetes_df$Income)
) {
  new_data <- tibble(
    BMI         = as.numeric(BMI),
    Age         = factor(Age, levels = Age_levels),
    GenHlth     = factor(GenHlth, levels = GenHlth_levels),
    HighBP      = factor(HighBP, levels = HighBP_levels),
    HighChol    = factor(HighChol, levels = HighChol_levels),
    PhysActivity = factor(PhysActivity, levels = PhysActivity_levels),
    DiffWalk    = factor(DiffWalk, levels = DiffWalk_levels),
    Sex         = factor(Sex, levels = Sex_levels),
    Education   = factor(Education, levels = Education_levels),
    Income      = factor(Income, levels = Income_levels)
  )
  class_pred <- predict(final_model, new_data, type = "class")
  prob_pred  <- predict(final_model, new_data, type = "prob")
  
  list(
    predicted_class = class_pred$.pred_class[1],
    probabilities   = prob_pred[1, ]
  )
}
# Example calls:
# http://localhost:PORT/pred?BMI=50&Age=50–54&GenHlth=Poor&HighBP=Yes%20High%20BP&HighChol=Yes%20High%20Chol&PhysActivity=NoActivity&DiffWalk=No%20Difficulty&Sex=Female&Education=Some%20college/technical%20school&Income=$50,000–<$75,000
# http://localhost:PORT/pred?BMI=32&Age=55–59&GenHlth=Fair&HighBP=High%20BP&HighChol=High%20Chol&PhysActivity=No%20Activity&DiffWalk=Difficulty%20Walking&Sex=Female&Education=College%20graduate&Income=$35,000–<$50,000
# http://localhost:PORT/pred?BMI=22&Age=25–29&GenHlth=Excellent&HighBP=No%20High%20BP&HighChol=No%20High%20Chol&PhysActivity=Activity&DiffWalk=No%20Difficulty&Sex=Male&Education=Some%20college/technical%20school&Income=$50,000–<$75,000

#* Info endpoint
#* @get /info
function() {
  list(
    name = "Bailey Allison",
    github_pages = "https://YOUR-USERNAME.github.io/YOUR-REPO-NAME/"
  )
}

#* Confusion matrix plot endpoint
#* @get /confusion
#* @png
function() {
  # Get predicted classes for the whole dataset
  preds <- predict(final_model, diabetes_df, type = "class")
  
  # Build a data frame with truth and predictions
  results <- tibble(
    truth = diabetes_df$Diabetes_012,
    .pred_class = preds$.pred_class
  )
  
  # Compute confusion matrix
  cm <- yardstick::conf_mat(results, truth = truth, estimate = .pred_class)
  
  # Plot confusion matrix
  autoplot(cm)
}
