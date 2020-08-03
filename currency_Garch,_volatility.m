%%%%%%% 3 separate tasks - checking which factors account for currency
%%%%%%% excess returns, simulating Garch processes, testing volatility
%%%%%%% trading. 

%% Explaining currency excess returns 1 %%
load('data_currency.mat'); 
%Log returns and getting difference of these for i*-i
f = log(F);
s = log(S);
f_s = f-s;

%R_x_t1 = (F./lagmatrix(S,-1))-1;
%This part is used to omit specific observations which do not have relevant
%return observations. I calculate nan_matrix if Ft/St+1 returns nan value.
%This can be the case if we have currency data on time t but don't have on
%t+1 or vice versa. 
idxS = ~isnan(S); %This is not changed, I use it for consistency.
lag_idxS = lagmatrix(idxS,0);
lag_idxS(lag_idxS == 0) = NaN;

idxF = ~isnan(F);
lag_idxF = lagmatrix(idxF,1); %bringing Ft to t+1. 
lag_idxF(lag_idxF == 0) = NaN;
%creating a nan matrix for observations where currency returns could not be
%calculated
nan_matrix = (lag_idxF./lag_idxS);
% I use the nan_matrix to filter out the observations at points where we
% cannot calculate currency returns so that we do not pick the specific
% currency into portfolio formation at a specific point.
f_s_filtered = f_s.*nan_matrix;

% constructing returns for each currency and then forming portfolio returns
% within for loops 
currency_returns = (lagmatrix(F,1)./S)-1;

% forming average portfolios based on interest rate differentials and their
% mean returns
for i = 1:434
    %firstly forming portfolios based on 30th percentile
    select_cols = f_s_filtered(i,:);
    pquint = quantile(select_cols,0.3,2);
    idx = (select_cols < pquint);
    avg_interest_30(i,1) = mean(select_cols(idx));
    %then indicating which returns on which currencies belong to this
    %portfolio
    select_currencies = currency_returns(i,:);
    returns_portfolio_30(i,1) = mean(select_currencies(idx));
end
%Removing NaN first row
r_X_L = returns_portfolio_30(2:end,1);

%repeating the same for 0.7 quintile
for i = 1:434
    %firstly forming portfolios based on 70th percentile
    select_cols1 = f_s_filtered(i,:);
    pquint1 = quantile(select_cols1,0.7,2);
    idx1 = (select_cols1 > pquint1);
    int_port_mean_70(i,1) = mean(select_cols1(idx1));
    %then indicating which returns on which currencies belong to this
    %portfolio
    select_currencies1 = currency_returns(i,:);
    returns_portfolio_70(i,1) = mean(select_currencies1(idx1));
end
%Removing NaN first row
r_X_H = returns_portfolio_70(2:end,1);

%Constructing a carry factor
HML = r_X_H - r_X_L;

%Constructing DOL - average excess returns for each currency
DOL = nanmean(currency_returns,2);
DOL = DOL(2:end,1);

matrix_factors = [HML';DOL']';
%Proceeding with fama-macbeth regression

%creating 5 portfolios for test assets
for i = 1:434
    %firstly forming portfolios based on 70th percentile
    select_portfolios = f_s_filtered(i,:);
    quint1 = quantile(select_portfolios,0.2,2);
    idxp1 = (select_portfolios <= quint1);
    quint2 = quantile(select_portfolios,0.4,2);
    idxp2 = (quint2 >= select_portfolios > quint1);
    quint3 = quantile(select_portfolios,0.6,2);
    idxp3 = (quint3 >= select_portfolios > quint2);
    quint4 = quantile(select_portfolios,0.8,2);
    idxp4 = (quint4 >= select_portfolios > quint3);
    idxp5 = (quint4 < select_portfolios);
    select_returns = currency_returns(i,:);
    port1(i,1) = nanmean(select_returns(idxp1),2);
    port2(i,1) = nanmean(select_returns(idxp2),2);
    port3(i,1) = nanmean(select_returns(idxp3),2);
    port4(i,1) = nanmean(select_returns(idxp4),2);
    port5(i,1) = nanmean(select_returns(idxp5),2);
end

%Converging these portfolios to a matrix
matrix_5port = [port1'; port2'; port3'; port4'; port5']';
matrix_5port = matrix_5port(2:end,:);

%FAMA-MACBETH REGRESSION
%setting up matrices
[dT, dN] = size(matrix_5port);
%size vectors
valpha = ones(dN,1);
vlambda = ones(dT,2);
%vectors for excess returns and residuals
mresid = ones(dT,dN);

%Time series regression
matrix_estimates_ts = [ones(dT,1) matrix_factors]\matrix_5port;
mresid = matrix_5port - [ones(dT,1) matrix_factors] * matrix_estimates_ts;
matrix_betas = matrix_estimates_ts(2:3,:);
%Cross sectional regression
vlambda = [matrix_betas']\matrix_5port';
alpha_cs = matrix_5port' - [matrix_betas']*vlambda;
mean_lambdas = mean(vlambda');

% Significance estimates and chi square
a = mean(alpha_cs,2);
transpose_alpha_cs = alpha_cs';
cov_alphas = cov(transpose_alpha_cs)/433;
chi_estimate = a'*pinv(cov_alphas)*a;
chi_critical = chi2inv(0.95,3);

%Descriptive statistis to present the results
mean_5port = mean(matrix_5port);
mean_factors = mean(matrix_factors);
%S.E.
%for lambda;
se_lambda = std(vlambda')./(433^0.5);
 
%for alpha_cs:
mean_alpha_cs = mean(alpha_cs');
se_alpha = std(alpha_cs')/(433^0.5);
sd_alpha = std(alpha_cs');

%clearvars -except mean_alpha_cs se_alpha sd_alpha mean_lambdas se_lambda mean_5port mean_factors chi_estimate chi_critical matrix_estimates_ts vlambda alpha_cs matrix_estimates_ts mresid matrix_betas matrix_5port DOL HML r_X_H r_X_L currency_returns f_s = f-s f_s_filtered f s currency_returns;
%% Garch processes %%
addpath(genpath(cd)) 
Data_m = xlsread('data_daily.xls');
Data_pick = Data_m(:,2:7);
Index_pick = Data_m(:,9);
%Calculating log returns:
Log_Ret = [log(Data_pick+1) log(Index_pick+1)];
Log_Ret_2 = Log_Ret.^2;
Log_Ret_SP = Log_Ret(:,end); %Log returns S&P500
Log_Ret_SP_2 = Log_Ret_2(:,end); %Squared log returns S&P500

%selecting p and q to minimize AIC or BIC
%firstly looking in the models with different p and q
warning('off','all')
options = optimset('MaxFunEvals', 20000000, ...
    'FunValCheck', 'off','Display', 'off', 'GradObj', 'off', 'Algorithm', 'sqp',...
    'Hessian', 'off', 'MaxIter', 800, 'TolFun',10^-8, 'TolX', 10^-8);
[param11, llk11, ht11, ~, robustSE11] = garchpq(Log_Ret_SP,1, 1,[],options);
[param21, llk21, ht21, ~, robustSE21] = garchpq(Log_Ret_SP,2, 1,[param11(1);param11(2);.01;param11(3)],options);
[param12, llk12, ht12, ~, robustSE12] = garchpq(Log_Ret_SP,1, 2,[param11;0.01],options);
[param22, llk22, ht22, ~, robustSE22] = garchpq(Log_Ret_SP,2, 2,[param12(1:2);0.01;param12(3:4)],options);
[param13, llk13, ht13, ~, robustSE13] = garchpq(Log_Ret_SP,1, 3,[param12;0.01],options);
[param31, llk31, ht31, ~, robustSE31] = garchpq(Log_Ret_SP,3, 1,[param21(1:3);0.01;param21(4)],options);
[param23, llk23, ht23, ~, robustSE23] = garchpq(Log_Ret_SP,2, 3,[param22;0.01],options);
[param32, llk32, ht32, ~, robustSE32] = garchpq(Log_Ret_SP,3, 2,[param22(1:3);0.01;param22(4:5)],options);
[param33, llk33, ht33, ~, robustSE33] = garchpq(Log_Ret_SP,3, 3,[param23(1:3);0.01;param23(4:6)],options);

% BIC criteria:
T = 6301; 
% formula: −2 log L/T + k log T/T;
BIC11 = -2*llk11/T+size(param11,1)*log(T)/T;
BIC12 = -2*llk12/T+size(param12,1)*log(T)/T;
BIC13 = -2*llk13/T+size(param13,1)*log(T)/T;
BIC21 = -2*llk21/T+size(param21,1)*log(T)/T;
BIC22 = -2*llk22/T+size(param22,1)*log(T)/T;
BIC23 = -2*llk23/T+size(param23,1)*log(T)/T;
BIC33 = -2*llk33/T+size(param33,1)*log(T)/T;
BIC31 = -2*llk31/T+size(param31,1)*log(T)/T;
BIC32 = -2*llk32/T+size(param32,1)*log(T)/T;
%GARCH(2,1) minimize the BIC
%%Descriptive statistics for GARCH(2,1)
disp(param21);
SE21 = diag(robustSE21);
disp('t-stats for GARCH(2,1): omega, alpha and beta');
tstat21 = param21./sqrt(diag(robustSE21));

%applying the model to SP500
[T, SP500]= size(Log_Ret_SP);
h = zeros(T,1);
constant = ((T-1)/2)*log(2*pi); %constant from the formula to calculate log-likelihood
coeff = [0.0001 0.0001 0.0001 0.0001];

%Making h(1) equal to unconditional variance
h(1) = coeff(1)/(1-(coeff(2)+coeff(3)*0.5+coeff(4)));
%Making a list for squared log-returns if log returns are negative -
%necessary for I calculation in the formula.
idx = Log_Ret_SP < 0;
Log_Ret_neg = Log_Ret_SP_2.* idx;

llk = loglikelihood(coeff,Log_Ret_SP_2,T,constant,Log_Ret_neg,h);
stat_cond = constraints_llk(coeff);

%just calling the functions for log-likelihood and stationarity(plus
%additional constraints)
functionllk = @(coeff)loglikelihood(coeff,Log_Ret_SP_2,T,constant,Log_Ret_neg,h);
functionstat_con = @(coeff)constraints_llk(coeff);

%coefficients maximising log-likelihood
opt_coeff = fmincon(functionllk, coeff,[],[],[],[],[],[],functionstat_con,[]);


%% Testing whether volatility is predictable %%
%Importing FF monthly data
zipname = importFrenchData(); % https://github.com/okomarov/importFrenchData
outdir  = [cd,'/'];
mret_FF = importFrenchData('F-F_Research_Data_Factors_TXT.zip',outdir);
idx         = mret_FF.Date >=192607 & mret_FF.Date <= 201812;
mret_FF     = mret_FF{idx,2:end}/100;
vrf         = mret_FF(:,end);
mret_FF(:,end) = []; 
MKT = mret_FF(:,1);
SMB = mret_FF(:,2);
HML = mret_FF(:,3);

%empty vector
weight = zeros(1,3);
y = Sharpe(MKT, SMB, HML, weight);
function_Sharpe = @(weight)Sharpe(MKT, SMB, HML, weight);
opt_weights_Sharpe = fminunc(function_Sharpe, weight);
% it does not use weight 1 so we construct new vector:
MVE_weights = [1-opt_weights_Sharpe(2)-opt_weights_Sharpe(3); opt_weights_Sharpe(2); opt_weights_Sharpe(3)];

% Using daily data
zipname = importFrenchData(); % https://github.com/okomarov/importFrenchData
outdir  = [cd,'/'];
mret_FF_Daily = importFrenchData('F-F_Research_Data_Factors_daily_TXT.zip',outdir);
idx         = mret_FF_Daily.Date >=19260701 & mret_FF_Daily.Date <= 20181231;
mret_FF_Daily     = mret_FF_Daily{idx,2:end-1}/100;

Moving_avg_F = movmean(mret_FF_Daily, [21,0]);
Empty = nan(24391,3);
Empty(22:22:end,:) = Moving_avg_F(22:22:end,:);
FF_22dayavg = fillmissing(Empty,'next');
Daily_FF = mret_FF_Daily - FF_22dayavg; %Difference between daily FF returns and 22 day avg
Daily_F2 = Daily_FF.^2;

Daily_F2_movsum = movsum(Daily_F2,[21 0]);
F_RV2 = Daily_F2_movsum(22:22:end,:); % this is the variance of FF 3 factors


%Further constructing RV(fmve)
%firstly constructing variance estimates
cov_mktsmb = Daily_FF(:,1) .* Daily_FF(:,2);
cov_mkthml = Daily_FF(:,1) .* Daily_FF(:,3);
cov_smbhml = Daily_FF(:,2) .* Daily_FF(:,3);
cov_FF_daily = [cov_mktsmb cov_mkthml cov_smbhml];
cov_FF = movsum(cov_FF_daily, [21,0]);
cov_FF = cov_FF(22:22:end,:);

[dN, dT] = size(F_RV2);

% this is a for loop designed to calculate the MVE portfolio variance for
% each row based on the factor weights and the covariances between the
% factors
for i = 1:dN
    Matrix = [F_RV2(i,1) cov_FF(i,1) cov_FF(i,2); 
              cov_FF(i,1) F_RV2(i,2) cov_FF(i,3);
              cov_FF(i,2) cov_FF(i,3) F_RV2(i,3)];
    MVE_RV(i,1) = MVE_weights'*Matrix*MVE_weights;
end
mean_MVE_RV = mean(MVE_RV);

% finding c which produces same unconditional variance
%calculating portfolio returns using daily data
F_MVE = mret_FF*MVE_weights;
F_MVE = F_MVE(1:end-2,1);
%summing daily returns over 22 days to get monthly returns
ff_mve = movsum(mret_FF_Daily, [21,0]);
ff_mve = ff_mve(22:22:end,:);
%setting initial c to a very low number
c = 0.0001;
%calling functions
varman = variance_managed_mve(F_MVE, MVE_RV,c);
constr = var_equal(F_MVE, MVE_RV,c); 
fun_varmanaged = @(c)variance_managed_mve(F_MVE, MVE_RV,c);
constr_var_equal = @(c)var_equal(F_MVE, MVE_RV,c);
%calculating optimal constant
c_optimal = fmincon(fun_varmanaged, c,[],[],[],[],[],[],constr_var_equal,[]);

unmanaged_var = var(F_MVE); 
managed_ret = (c_optimal./MVE_RV).*F_MVE;
managed_var = var(managed_ret);
Sharpe_nonmanaged = mean(F_MVE)/sqrt(unmanaged_var);  %This is monthly!
Sharpe_managed = mean(managed_ret)/sqrt(managed_var); %This is monthly!

%fitting linear model
mb1 = [ones(1108,1) F_MVE]\managed_ret;
mresid1 = managed_ret - [ones(1108,1) F_MVE] * mb1;
%regstats(managed_ret,F_MVE,'linear') %can also use this to infer the
% significance of coefficients

clearvars -except MVE_weights F_RV2 cov_FF MVE_RV ff_mve fun_varmanaged constr_var_equal c_optimal unmanaged_var managed_ret managed_var Sharpe_nonmanaged Sharpe_managed mb1 mean_MVE_RV Log_Ret_SP opt_coeff Log_Ret_SP_2 llk11 llk12 llk13 llk21 llk22 llk23 llk33 llk31 llk32 Log_Ret_neg h(1) param21 SE21 tstat21 BIC11 BIC12 BIC13 BIC21 BIC22 BIC23 BIC33 BIC31 BIC32 mean_alpha_cs se_alpha sd_alpha mean_lambdas se_lambda mean_5port mean_factors chi_estimate chi_critical matrix_estimates_ts vlambda alpha_cs matrix_estimates_ts mresid matrix_betas matrix_5port DOL HML r_X_H r_X_L currency_returns f_s = f-s f_s_filtered f s currency_returns;