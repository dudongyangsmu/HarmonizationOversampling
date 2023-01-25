%% Model building
% Comparisons of harmonization and oversampling methods via different machine learning classifiers.
% data needs be splitted by k-fold cross validation * n times.
% Additionly, 5-fold cross validation within the training data was used to determine the
% number of selected features, then using all training data (with selected features) to build model.

close all; clear all; clc;

addpath(genpath('G:\ADC_SCC\HarmOversamp\Upload\Oversampling\SASYNO'))
addpath(genpath('G:\ADC_SCC\HarmOversamp\Upload\Oversampling\github_repo'))

savePath = 'G:\ADC_SCC\HarmOversamp\Upload\';
harPath = 'G:\ADC_SCC\HarmOversamp\Upload\HarmonizedData\';

for iter = 1  % outer repeated 5 times, iter defines the times of external testing, eg., iter=1:5.

    HarFeaPath = [harPath,'DataDivision',num2str(iter),'\']; 
    dataPath = ['G:\ADC_SCC\HarmOversamp\Upload\','DataDivision',num2str(iter),'\'];

    % load original data
    load([dataPath,'data_internal.mat']); % loading Data_internal and Label_internal
    Label_internal = Label_internal+1;
    load([dataPath,'data_external.mat']); % loading Data_external and Label_external
    Label_test = Label_external+1;

    % load internal partition
    load([dataPath,'data_in_partition']);

    for R = 1:5   % here R means internal cross validation runs
        idxTrain = training(in_partition,R);
        idxValid = test(in_partition,R);
        %% Harmonization
        for H = 1:5 % here H means select different harmonization methods
            % selection of the harmonization methods
            listHarmonization = {'None','combat_v1','centering','SVD_v2','ICA 5_v2'};
            harmonization_method = listHarmonization{H};
            disp(['Harmonization Method is',' ',harmonization_method]);
            switch harmonization_method
                case "None"
                    data = Data_internal;
                    exData = Data_external;
                case "combat_v1"
                    load([HarFeaPath,'combat_v1.mat']);
                    data = double(FeatureValue);  clear FeatureValue;
                    load([HarFeaPath,'TE_combat_v1.mat']);
                    exData = double(FeatureValue);  clear FeatureValue;
                case "centering"
                    load([HarFeaPath,'centering.mat']);
                    data = double(FeatureValue);  clear FeatureValue;
                    load([HarFeaPath,'TE_centering.mat']);
                    exData = double(FeatureValue);  clear FeatureValue;
                case "SVD_v2"
                    load([HarFeaPath,'SVD_v2.mat']);
                    data = double(FeatureValue);  clear FeatureValue;
                    load([HarFeaPath,'TE_SVD_v2.mat']);
                    exData = double(FeatureValue);  clear FeatureValue;
                case "ICA 5_v2"
                    load([HarFeaPath,'ICA 5_v2.mat']);
                    data = double(FeatureValue);  clear FeatureValue;
                    load([HarFeaPath,'TE_ICA 5_v2.mat']);
                    exData = double(FeatureValue);  clear FeatureValue;
            end

           %% divide tr/te
            label = Label_internal;
            exLabel = Label_test;
            
            F_tr = data(idxTrain,:);
            F_te = data(idxValid,:);
            label_tr = label(idxTrain);
            label_te = label(idxValid);

          %% feature data preprocessing
            % training data preprocessing
            % remove NaN Inf
            mask1 = isinf(F_tr);F_tr(mask1) = 1000; clear mask1;
            mask2 = isnan(F_tr);F_tr(mask2) = 0; clear mask2;
            % feature normalization--train
            mean_tr=mean(F_tr);
            std_tr=std(F_tr);
            F_tr = zscore(F_tr);
            % validtion feature data preprocessing
            % remove NaN Inf
            mask1 = isinf(F_te);F_te(mask1) = 1000; clear mask1;
            mask2 = isnan(F_te);F_te(mask2) = 0; clear mask2;
            mask1 = isinf(exData);exData(mask1) = 1000; clear mask1;
            mask2 = isnan(exData);exData(mask2) = 0; clear mask2;
            % normalization
            for n=1:size(F_te,1)
                F_te(n,:) = (F_te(n,:)-mean_tr)./std_tr;
            end
            for n=1:size(exData,1)
                exData(n,:) = (exData(n,:)-mean_tr)./std_tr;
            end

            for A = 1:6    % here A means different imbalanced adjustment methods
                % parameters of oversampling
                k = 5; % define the number of neighbors to consider
                adds = numel(find(label_tr==2))-numel(find(label_tr==1));
                num2Add = [0,adds]; %
              %% data augmentation
                listAugmentation = {'None','SMOTE','ADASYN','B_SMOTE','S_SMOTE','SASYNO'};
                augmentation_method = listAugmentation{A};
                disp(['Imbalanced Method is',' ',augmentation_method]);
                if ~strcmp(augmentation_method,'None')&&~strcmp(augmentation_method,'SASYNO')
                    dataset = array2table(F_tr);
                    dataset = addvars(dataset, string(label_tr),...
                        'NewVariableNames','label');
                    labels = dataset(:,end);
                    t = tabulate(dataset.label);
                    uniqueLabels = string(t(:,1));
                    % labelCounts = cell2mat(t(:,2));
                    newdata = table; % creat am empty dable
                    visdataset = cell(length(uniqueLabels),1);
                    % for each class
                    for ii=1:length(uniqueLabels)
                        switch augmentation_method
                            case "SMOTE"
                                [tmp,visdata] = mySMOTE(dataset,uniqueLabels(ii),num2Add(ii),...
                                    "NumNeighbors",k, "Standardize", false);
                            case "ADASYN"
                                [tmp,visdata]  = myADASYN(dataset,uniqueLabels(ii),num2Add(ii),...
                                    "NumNeighbors",k, "Standardize", false);
                            case "B_SMOTE"
                                [tmp,visdata] = myBorderlineSMOTE(dataset,uniqueLabels(ii),num2Add(ii),...
                                    "NumNeighbors",k, "Standardize", false);
                            case "S_SMOTE"
                                [tmp,visdata] = mySafeLevelSMOTE(dataset,uniqueLabels(ii),num2Add(ii),...
                                    "NumNeighbors",k, "Standardize", false);
                        end
                        newdata = [newdata; tmp];
                        visdataset{ii} = visdata;
                    end
                    generateData = table2array(newdata(:,1:end-1));
                    generateLabel = table2array(newdata(:,end));
                    generateLabel = str2double(generateLabel);
                    F = [F_tr;generateData];
                    Label = [label_tr;generateLabel];
                elseif strcmp(augmentation_method,'SASYNO')
                    [F,Label]=SASYNO(F_tr,label_tr-1);
                elseif strcmp(augmentation_method,'None')
                    F = F_tr;
                    Label = label_tr;
                end


              %% feature selection
                % mrmr
                [idx,scores] = fscmrmr(F, Label);
                % --------perform the 5 fold CV within the training data to
                % determine the number of selected features
                for numF = 2:30   % number of selected features by mrmr
                    for f = 1:5  % here f means 5-fold cross validation within training set to turn parameters
                        rng(0);
                        tr_partition = cvpartition(Label,'KFold',5);
                        idx_train_cv = training(tr_partition,f);
                        idx_valid_cv = test(tr_partition,f);
                        F_tr_cv = F(idx_train_cv,:);
                        label_tr_cv = Label(idx_train_cv);
                        F_valid_cv = F(idx_valid_cv,:);
                        label_valid_cv = Label(idx_valid_cv);
                        index_MR = idx(1:numF);
                        % valid_model=TreeBagger(140,F_tr_cv(:,index_MR),label_tr_cv);
                        % [~,valid_prob] =  predict(valid_model,F_valid_cv(:,index_MR));
                        % valid_model = mnrfit(F_tr_cv(:,index_MR),label_tr_cv);
                        % valid_prob = mnrval(valid_model,F_valid_cv(:,index_MR));
                        valid_model=fitcdiscr(F_tr_cv(:,index_MR),label_tr_cv);
                        [~,valid_prob] = predict(valid_model,F_valid_cv(:,index_MR));
                        % mask3 = isnan(valid_prob(:,2));
                        % valid_prob(mask3,2) = 1 - valid_prob(mask3,1);
                        [~,~,~,valid_AUC,~] = perfcurve(label_valid_cv, valid_prob(:,2),2);
                        valid_AUC_cv(numF,f)= valid_AUC;
                    end
                end

                [~,topF]= max(mean(valid_AUC_cv,2)); clear valid_AUC_cv;
                index_MR = idx(1:topF);
                scores_o = scores(idx(1:topF));

              %%  built model and valid
                for C=1:5
                    listclassifier = {'LR','LDA','NB','RF','SVM'};
                    classification_method = listclassifier{C};
                    disp(['Classification Method is',' ',classification_method]);
                    switch classification_method
                        case "LR"
                            model = mnrfit(F(:,index_MR),Label);
                            prob_te = mnrval(model,F_te(:,index_MR));
                            prob_te_ex = mnrval(model,exData(:,index_MR));
                        case "LDA"
                            model=fitcdiscr(F(:,index_MR),Label);
                            [~,prob_te] = predict(model,F_te(:,index_MR));
                            [~,prob_te_ex] = predict(model,exData(:,index_MR));
                        case "RF"
                            rng(1);
                            model=TreeBagger(140,F(:,index_MR), Label,'OOBPrediction','On','OOBPredictorImportance','On');
                            [~,prob_te] =  predict(model,F_te(:,index_MR));
                            [~,prob_te_ex] = predict(model,exData(:,index_MR));
                        case "SVM"
                            rng(1);
                            model=fitcsvm(F(:,index_MR), Label,'KernelFunction','linear','KernelScale','auto');
                            [~,prob_te] =  predict(model,F_te(:,index_MR));
                            [~,prob_te_ex] = predict(model,exData(:,index_MR));   
                    end


                 %% results of internal validation data
                    threshold = 0.5;
                    [teo_X,teo_Y,~,AUC_teo,~] = perfcurve(label_te, prob_te(:,2),2);
                    predict_label_teo=zeros(size(label_te));
                    for j=1:size(label_te)
                        if prob_te(j,2) > threshold
                            predict_label_teo(j)=2;
                        else
                            predict_label_teo(j)=1;
                        end
                    end
                    %ACC_teo = sum(predict_label_teo==label_te)/size(label_te,1);
                    tp_teo =   sum(predict_label_teo(predict_label_teo==label_te) ==2);      
                    fn_teo =   sum(predict_label_teo(predict_label_teo~=label_te) ==1);       
                    fp_teo =   sum(predict_label_teo(predict_label_teo~=label_te) ==2);        
                    tn_teo =   sum(predict_label_teo(predict_label_teo==label_te) ==1);        
                    SEN_teo = tp_teo/(tp_teo+fn_teo);
                    SPE_teo= tn_teo/(tn_teo+fp_teo);
                    % compute the balanced accuracy
                    BACC_teo = (SEN_teo + SPE_teo)/2;
                    gmean_teo = sqrt((tp_teo/(tp_teo+fn_teo))*(tn_teo/(tn_teo+fp_teo)));
                    % precision_teo = tp_teo/(tp_teo+fp_teo);
                    % recall_teo = tp_teo/(tp_teo+fn_teo);
                    % f1_teo = (2*precision_teo*recall_teo)/(precision_teo+recall_teo);

                 %% results of external validation data
                    prob_te_ex(:,2) = 1-prob_te_ex(:,1);
                    [x_ex,y_ex,~,AUC_ex,~] = perfcurve(exLabel, prob_te_ex(:,2),2);
                    predict_label_ex=zeros(size(exLabel));
                    for j=1:size(exLabel)
                        if prob_te_ex(j,2) > threshold
                            predict_label_ex(j)=2;
                        else
                            predict_label_ex(j)=1;
                        end
                    end
                    %ACC_ex = sum(predict_label_ex==exLabel)/size(exLabel,1);
                    tp_ex =   sum(predict_label_ex(predict_label_ex==exLabel) ==2);        
                    fn_ex =   sum(predict_label_ex(predict_label_ex~=exLabel) ==1);         
                    fp_ex =   sum(predict_label_ex(predict_label_ex~=exLabel) ==2);        
                    tn_ex =   sum(predict_label_ex(predict_label_ex==exLabel) ==1);   
                    SEN_ex = tp_ex/(tp_ex+fn_ex);
                    SPE_ex= tn_ex/(tn_ex+fp_ex);
                    % compute the balanced accuracy
                    BACC_ex = (SEN_ex + SPE_ex)/2;
                    gmean_ex = sqrt((tp_ex/(tp_ex+fn_ex))*(tn_ex/(tn_ex+fp_ex)));
                    % precision_ex = tp_ex/(tp_ex+fp_ex);
                    % recall_ex = tp_ex/(tp_ex+fn_ex);
                    % f1_ex = (2*precision_ex*recall_ex)/(precision_ex+recall_ex);


                  %% save result
                    stats(C).harmonization_method = harmonization_method;
                    stats(C).augmentation_method = augmentation_method;
                    stats(C).classification_method = classification_method;

                    stats(C).AUC_valid = AUC_teo;
                    stats(C).BACC_valid = BACC_teo;
                    stats(C).SEN_teo= SEN_teo;
                    stats(C).SPE_teo= SPE_teo;
                    stats(C).gmean_teo= gmean_teo;
                    % stats(C).precision_teo= precision_teo;
                    % stats(C).recall_teo= recall_teo;
                    % stats(C).f1_teo= f1_teo;

                    stats(C).prob_teo = prob_te;
                    stats(C).label_te = label_te;
                    stats(C).teo_X = teo_X;
                    stats(C).teo_Y = teo_Y;

                    stats(C).AUC_ex = AUC_ex;
                    stats(C).BACC_ex = BACC_ex;
                    stats(C).SEN_ex= SEN_ex;
                    stats(C).SPE_ex= SPE_ex;
                    stats(C).gmean_ex = gmean_ex;
                    % stats(C).precision_ex= precision_ex;
                    % stats(C).recall_ex= recall_ex;
                    % stats(C).f1_ex= f1_ex;


                    stats(C).prob_te_ex = prob_te_ex;
                    stats(C).exLabel = exLabel;
                    stats(C).x_ex = x_ex;
                    stats(C).y_ex = y_ex;

                end

                fileName=[savePath,'SimpleResults\','ex_',num2str(iter),'th\','BatchName\',harmonization_method,'\',augmentation_method,'\']; % modify!! Batch name,e.g.,center,scanner,reconstruction.
                if ~exist(fileName)
                    mkdir(fileName);
                end
                save([fileName,num2str(R),'-','stats.mat'],'stats');
                clear stats;
                % save aug data
                if ~strcmp(augmentation_method,'None')
                    aug.generateData = generateData;
                    aug.generateLabel = generateLabel;
                    save([fileName,num2str(R),'-','aug.mat'],'aug'); clear aug;
                end

            end
        end
    end
end
toc







