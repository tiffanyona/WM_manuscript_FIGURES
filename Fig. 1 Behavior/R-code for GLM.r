library(readr)
library(lme4)
library(glmnet)
library(broom)
library(glmmLasso)


animal_list = c('E04', 'E08', 'E10',
                'E05', 'E03', 'E07', 'E06', 'E11', 'E13', 'E14', 'E09', 'E12',
                'E21', 'E22', 'E16', 'E20', 'E17', 'E19', 'E18', 'E15','C12', 'C13', 'C10b', 'C15', 'C18', 'C19', 'C20', 'C22', 'N08',
                'N13', 'N07', 'N09', 'N11', 'N03', 'N04', 'N02', 'N05', 'C28',
                'C32', 'C34', 'C39', 'C37', 'C38', 'C36', 'N19', 'N27', 'N24',
                'N20', 'N22', 'N28', 'N26', 'N21', 'N25')

# animal_list= c("E03", "E04", "E05_3", "E05_10", "E06", "E07_3", "E07_10",
#               "E08", "E09", "E10", "E11", "E12_3", "E12_10", "E13", "E14",
#               "E15_3", "E15_10", "E16_3", "E16_10", "E17_3", "E17_10", "E18",
#               "E19", "E20_3", "E20_10", "E21", "E22", "N02", "N03", "N04", "N05_3",
#               "N05_10", "N07_3", "N07_10", "N08", "N09", "N11_3", "N11_10", "N13", "N19",
#               "N20", "N21", "N22", "N24_3", "N24_10", "N25_3", "N25_10", "N26", "N27_3",
#               "N27_10", "N28_3", "N28_10", "C10b", "C12", "C13", "C15", "C18", "C19", "C20",
#               "C22", "C28_3", "C28_10",  "C32", "C34", "C36", "C37_3", "C37_10", "C38", "C39")

formula = 'vector_answer ~ SL + SR + SL:D + SR:D + D:exp_C + SL:T + SR:T + SL:D:T + SR:D:T + T:exp_C + T:D:exp_C + exp_C - 1 +  (exp_C + SL + SR + T|session) '
# formula = 'vector_answer ~ SL + SR + SL:D + SR:D + D:exp_C + SL:T + SR:T + SL:D:T + SR:D:T + T:exp_C + T:D:exp_C + exp_C - 1 +  (1|session) '
# formula = 'vector_answer ~ SL + SR + SL:D_norm + SR:D_norm + D_norm:exp_C + SL:T + SR:T + SL:D_norm:T + SR:D_norm:T + T:exp_C + T:D_norm:exp_C + exp_C - 1 +  (1|session) '
# formula = 'vector_answer ~ SL + SR + SL:D + SR:D + D:exp_C + exp_C - 1 + (1|session) '
# formula = 'vector_answer ~ S + S:D + D '

listAIC = list()

for (b in animal_list) {
  print(b)
  
  # a= "C:\\Users\\Tiffany\\Google Drive\\WORKING_MEMORY\\PAPER\\Figures\\"
  a= "G:\\Mi unidad\\WORKING_MEMORY\\PAPER\\ANALYSIS_Figures\\"
  c="_x.csv"
  d="_y.csv"
  e="_all.csv"
  
  # c="_realx.csv"
  # d="_realy.csv"
  # e="_real.csv"

  # x = read_csv(paste(a, b, c, sep=""))
  # y = read_csv(paste(a, b, d, sep=""))
  # x = data.matrix(x)
  # y = data.matrix(y)
  # 
  
  # ----------------- GLM normal sin regularizar --------
  
  # df = read_csv(paste(a, b, e, sep=""))
  # df = as.data.frame(df)
  # 
  # fit = glm(formula, data=df, family= binomial(link='probit'))
  # print(summary(fit))
  # listAIC = append(listAIC, AIC(fit))
  # coeficients = coef
  
  # df = read_csv(paste(a, b, e, sep=""))
  # df = as.data.frame(df)
  # 
  # fit = glm(formula, data=df, family= binomial(link='probit'))
  # print(summary(fit))
  # listAIC = append(listAIC, AIC(fit))
  # coeficients = coef(summary(fit))
  
  # ------------------  GLM regularizando -------------------
  
  # cv_ridge = cv.glmnet(x, y, alpha = 0, family = binomial(link= "probit"), lambda=seq(from=0,to=0.3,length.out=500), standardize = FALSE, intercept=FALSE)
  # plot(cv_ridge, label=TRUE)
  # fit = glmnet(x, y, alpha = 0, family = binomial(link= "probit"), lambda=cv_ridge$lambda.min, trace.it = TRUE, intercept=FALSE, standardize = FALSE)
  # # plot(fit, label=TRUE)
  # print(cv_ridge$lambda.min)
  # coeficients = coef(fit)
  # print(coeficients)
  # 
  # tidy_lmfit = tidy(fit)
  # write.csv(tidy_lmfit, paste(a, b, "_result.csv", sep=""))
  # 
  # tLL <- -deviance(fit) # 2*log-likelihood
  # k = fit$df
  # n = fit$nobs
  # AICc = -tLL+2*k+2*k*(k+1)/(n-k-1)
  # print(AICc)
  # 
  # AIC <- -tLL+2*k
  # print(AIC)
  # 
  # listAIC = append(listAIC, AIC)
  
  # ---------- Same but for mixed models -----------
  df = read_csv(paste(a, b, e, sep=""))
  df = as.data.frame(df)
  
  fit = glmer(formula, data=df, family= binomial(link='probit'), control = glmerControl(calc.derivs = FALSE, optimizer="bobyqa"))
  #  fit = glmmLasso('vector_answer ~ S + S:D + D ', data=df, rnd = NULL)
  listAIC = append(listAIC, AIC(fit))
  coeficients = coef(summary(fit))
  write.csv(coeficients, paste(a, b, "_result_mixed_lasso.csv", sep=""))
  
}

# write.csv(listAIC, paste(a, "list_AIC_mixed.csv", sep=""))


