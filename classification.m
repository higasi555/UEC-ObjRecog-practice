classdef classification
    methods(Static)
        % 以前の課題で作成した関数（後方互換性のために維持）
        function nearest_neighbor(img_list, encoded_v, which_vector)
            fprintf('=== Nearest Neighbor for %s ===\n', which_vector);
            n = size(encoded_v,1); % 200

            correct_count = 0;
            for i = 1:n
                query = encoded_v(i,:);
                dist = sum((encoded_v - query).^2, 2);
                dist(i) = Inf;
                [~, idx_min] = min(dist);

                query_label = (i<=100);
                found_label = (idx_min<=100);
                if query_label == found_label
                    correct_count = correct_count + 1;
                end
            end
            accuracy = correct_count / n;
            fprintf('Accuracy: %.2f%%\n', accuracy*100);

            % 類似画像例を表示
            sample_indices = [1, 50, 101, 150, 200];
            for q = sample_indices
                if strcmp(which_vector, 'Hist64')
                    utils.show_nearest_image_color(encoded_v, img_list, q);
                else
                    % which_vector = 'BoF' の場合
                    utils.show_nearest_image_bof(encoded_v, img_list, q);
                end
            end
        end

        % 以前の課題で作成した関数（後方互換性のために維持）
        function svm(train_list, train_v, train_label, which_vector, test_list, test_v, test_label)
            fprintf('=== SVM classification for %s ===\n', which_vector);

            %% 1) 線形SVMで学習
            tic;
            model_linear = fitcsvm(train_v, train_label, 'KernelFunction', 'linear', 'Standardize', true);
            tTrain_linear = toc;

            % テストデータを分類
            tic;
            pred_label_linear = predict(model_linear, test_v);
            tTest_linear = toc;
            fprintf('[Linear] TrainTime=%.2f[s], TestTime=%.2f[s]\n', tTrain_linear, tTest_linear);
            
            % 精度計算
            test_correct_linear = sum(pred_label_linear == test_label);
            test_acc_linear = test_correct_linear / length(test_label) * 100;
            fprintf('[Liner] TestAcc=%.2f%%\n', test_acc_linear);


            %% 2) RBFカーネルで学習
            tic;
            model_rbf = fitcsvm(train_v, train_label, 'KernelFunction', 'rbf', 'KernelScale', 'auto', 'Standardize', true);
            tTrain_rbf = toc;

            % テストデータを分類
            tic;
            pred_label_rbf = predict(model_rbf, test_v);
            tTest_rbf = toc;
            fprintf('[RBF] TrainTime=%.2f[s], TestTime=%.2f[s]\n', tTrain_rbf, tTest_rbf);

            % 精度計算
            correct_rbf = sum(pred_label_rbf == test_label);
            acc_rbf = correct_rbf / length(test_label)*100;
            fprintf('[RBF] TestAcc=%.2f%%\n', acc_rbf);
        end

        % 以前の課題で作成した関数（後方互換性のために維持）
        function naive_bayes(train_list, train_v, train_label, which_vector, test_list, test_v, test_label)
            fprintf('=== Naive Bayes for %s ===\n', which_vector);
    
            % まず，各visual wordsの出現頻度を，ポジティブ画像，ネガティブ画像 それぞれ別々にカウント
            pos_v = train_v(train_label > 0, :);
            neg_v = train_v(train_label < 0, :);
    
            % ただし，出現頻度０になると計算できないので， すべてのvisual wordsの出現頻度に１を加える
            pr_pos = sum(pos_v, 1) + 1;
            pr_neg = sum(neg_v, 1) + 1;

            % そして，それぞれの総出現頻度で割って正規化して，logをとる
            pr_pos = pr_pos / sum(pr_pos);
            pr_neg = pr_neg / sum(pr_neg);

            pr_pos = log(pr_pos);
            pr_neg = log(pr_neg);
    
            %% 2) 次に，テスト画像を分類
            nTest = size(test_v, 1);
            pred_label = zeros(nTest, 1);
    
            for i = 1:nTest
                im = test_v(i, :);  % 1 x d

                % 未知画像のbag-of-featuresベクトルimに対応する
                % pr_pos, pr_neg の和をポジティブ，ネガティブそれぞれ求める

                max0 = max(im);
                idx = [];
                for c = 1:max0
                    idx = [idx, find(im >= c)];
                end

                pr_im_pos = sum(pr_pos(idx));
                pr_im_neg = sum(pr_neg(idx));

    
                % pr_im_pos と pr_im_negを比較して，大きい方に画像imを分類
                if pr_im_pos > pr_im_neg
                    pred_label(i) = +1;
                else
                    pred_label(i) = -1;
                end
            end
    
            % 精度計算
            correct = sum(pred_label == test_label);
            acc = correct / nTest * 100;
            fprintf('TestAcc=%.2f%%\n', acc);

        end

        % 以前の課題で作成した関数（後方互換性のために維持）
        function feature_maps(train_list, train_v, train_label, which_vector, test_list, test_v, test_label)
            fprintf('=== Feature Maps for %s ===\n', which_vector);

            % 写像
            train_v_mapped = classification.map_chi2(train_v);
            test_v_mapped  = classification.map_chi2(test_v);

            % svmで分類
            classification.svm(train_list, train_v_mapped, train_label, which_vector, test_list, test_v_mapped, test_label);

        end

        % 以前の課題で作成した関数（後方互換性のために維持）
        function naive_bayes_nn(train_list, train_label, test_list, test_label)
            fprintf('=== NBNN ===\n');
            
            tic;

            % すべてのポジティブ学習画像から特徴点を抽出します． それらをポジティブ特徴点集合とします．
            % 同様に すべてのネガティブ学習画像から特徴点を抽出して ネガティブ特徴点集合を作ります．
            posFeatures = [];
            negFeatures = [];
            for i = 1:length(train_list)
                I = imread(train_list{i});
                Igray = rgb2gray(I);

                pts = detectSURFFeatures(Igray);
                pts = pts.selectStrongest(1000);          % 上位1000点を使う
                [f, ~] = extractFeatures(Igray, pts);

                % 特徴点集合の作成
                if train_label(i) > 0
                    posFeatures = [posFeatures; double(f)];
                else
                    negFeatures = [negFeatures; double(f)];
                end
            end

            trainTime = toc;
            fprintf('TrainTime=%.2f[s]\n', trainTime);
            
            % 各テスト画像に対してNBNN分類
            tic;
            nTest = length(test_list);
            pred_label = zeros(nTest, 1);

            for i = 1:nTest
                I = imread(test_list{i});
                Igray = rgb2gray(I);

                % 次に分類したい画像(未知画像)から特徴点を抽出して，
                % このすべての特徴点について，最も近いベクトル(SIFT128次元ベクトル) をポジティブ特徴点集合から探します．
                pts_test = detectSURFFeatures(Igray);
                pts_test = pts_test.selectStrongest(1000);
                [f_test, ~] = extractFeatures(Igray, pts_test);
                f_test = double(f_test);

                % D{pos}, D{neg}の計算
                sumDistPos = 0;
                sumDistNeg = 0;

                for j = 1:size(f_test,1)
                    desc = f_test(j,:);  % 1×(64 or 128)

                    % そして， 各特徴点から最も近いポジティブ特徴点集合の特徴点までの
                    % ユークリッド距離をすべての点について合計します． これを D{pos}とします．
                    dist_pos = sum( (posFeatures - desc).^2, 2 );
                    dp = min(dist_pos);
                    sumDistPos = sumDistPos + dp;

                    % 同様にネガティブ特徴点集合の特徴点までの
                    % ユークリッド距離を分類したい画像の特徴点すべてについて求めて合計します．これをD{neg}とします．
                    dist_neg = sum( (negFeatures - desc).^2, 2 );
                    dn = min(dist_neg);
                    sumDistNeg = sumDistNeg + dn;
                end

                % そして最後に，D{pos}とD{neg}の比較を行います．
                % D{pos}が小さければ未知画像はポジティブ画像， D{neg}が小さければ未知画像はネガティブ画像に分類されます
                if sumDistPos < sumDistNeg
                    pred_label(i) = +1;
                else
                    pred_label(i) = -1;
                end
            end

            testTime = toc;
            fprintf('TestTime=%.2f[s]\n', testTime);

            % 精度計算
            correct = sum(pred_label == test_label);
            acc = correct / nTest * 100;
            fprintf('TestAcc=%.2f%%\n', acc);
        end

        % 以前の課題で作成した関数（後方互換性のために維持）
        function svm_kfold(all_data, all_label, which_vector, cv)
            fprintf('=== SVM K-fold (%d) for %s ===\n', cv, which_vector);

            % pos/neg分割
            idx_pos = find(all_label==+1);
            idx_neg = find(all_label==-1);

            data_pos = all_data(idx_pos,:);
            data_neg = all_data(idx_neg,:);

            n_pos = size(data_pos,1);
            n_neg = size(data_neg,1);

            % idxを用意
            idx_p = 1:n_pos;
            idx_n = 1:n_neg;

            accuracy_list = [];  % 各foldの精度を格納

            for i=1:cv
                train_pos = data_pos(mod(idx_p,cv)~=(i-1), :);  % 評価用以外
                eval_pos  = data_pos(mod(idx_p,cv)==(i-1), :);  % 評価用

                train_neg = data_neg(mod(idx_n,cv)~=(i-1), :);
                eval_neg  = data_neg(mod(idx_n,cv)==(i-1), :);

                train_data = [train_pos; train_neg];
                train_label= [ones(size(train_pos,1),1); -ones(size(train_neg,1),1)];
                eval_data  = [eval_pos;  eval_neg];
                eval_label = [ones(size(eval_pos,1),1);  -ones(size(eval_neg,1),1)];

                %fitcsvm
                model = fitcsvm(train_data, train_label, 'KernelFunction','linear','Standardize',true);

                %分類
                predicted_label = predict(model, eval_data);

                %精度計算
                correct = sum(predicted_label == eval_label);
                ac = correct / length(eval_label);
                accuracy_list(end+1) = ac;

                fprintf(' Fold %d: accuracy=%.2f%%\n', i, ac*100);
            end

            %平均精度
            mean_ac = mean(accuracy_list);
            fprintf('Mean accuracy = %.2f%%\n', mean_ac*100);
        end

        % linearかrbfでk-foldでsvmを学習・評価し、結果を出力する関数
        % 最終的に、すべてのデータで学習したモデルを返す
        function Return = svm_kfold_new(all_data, all_label, which_vector, cv, kernelType)
            % rbfを指定したならrbf、そうでないならlinear
            if strcmpi(kernelType, 'rbf')
                kernelType = 'rbf';
            else
                kernelType = 'linear';
            end

            fprintf('=== SVM K-fold (%d) for %s with kernel %s ===\n', cv, which_vector, kernelType);

            % 正例(+1)と負例(-1)に分割
            idx_pos = find(all_label == +1);
            idx_neg = find(all_label == -1);

            data_pos = all_data(idx_pos, :);
            data_neg = all_data(idx_neg, :);

            n_pos = size(data_pos, 1);
            n_neg = size(data_neg, 1);

            % 各foldごとにインデックスを用意
            idx_p = 1:n_pos;
            idx_n = 1:n_neg;

            accuracy_list = zeros(cv, 1);  % 各foldの精度を格納するリスト

            % k-fold
            for i = 1:cv
                % 各foldにおいて、正例と負例を分割
                train_pos = data_pos(mod(idx_p, cv) ~= (i-1), :);
                eval_pos  = data_pos(mod(idx_p, cv) == (i-1), :);

                train_neg = data_neg(mod(idx_n, cv) ~= (i-1), :);
                eval_neg  = data_neg(mod(idx_n, cv) == (i-1), :);

                % 学習用データとラベル
                train_data = [train_pos; train_neg];
                train_label = [ones(size(train_pos, 1), 1); -ones(size(train_neg, 1), 1)];
                % 評価用データとラベル
                eval_data = [eval_pos; eval_neg];
                eval_label = [ones(size(eval_pos, 1), 1); -ones(size(eval_neg, 1), 1)];

                % SVM
                model_cv = fitcsvm(train_data, train_label, 'KernelFunction', kernelType, 'Standardize', true);

                % 予測ラベル
                predicted_label = predict(model_cv, eval_data);

                % 精度計算
                correct = sum(predicted_label == eval_label);
                ac = correct / length(eval_label);
                accuracy_list(i) = ac;

                fprintf(' Fold %d: accuracy = %.2f%%\n', i, ac * 100);
            end

            % 平均精度
            mean_ac = mean(accuracy_list);
            fprintf('Mean accuracy = %.2f%%\n', mean_ac * 100);

            % 最終モデルの学習
            model = fitcsvm(all_data, all_label, 'KernelFunction', kernelType, 'Standardize', true);
            fprintf('Final model trained\n');

            predicted_label = predict(model, all_data);

            % 精度計算
            correct = sum(predicted_label == all_label);
            ac = correct / length(all_label);
            fprintf('Final: accuracy = %.2f%%\n', ac * 100);

            Return = model;
        end


        % linearかrbfでk-foldでsvmを学習・評価し、結果を出力する関数
        % 最終的に、foldの最終回で作ったモデルを返す（結果の考察用）
        function Return = svm_kfold_new_model5(all_data, all_label, which_vector, cv, kernelType)
            % rbfを指定したならrbf、そうでないならlinear
            if strcmpi(kernelType, 'rbf')
                kernelType = 'rbf';
            else
                kernelType = 'linear';
            end

            fprintf('=== SVM K-fold (%d) for %s with kernel %s ===\n', cv, which_vector, kernelType);

            % 正例(+1)と負例(-1)に分割
            idx_pos = find(all_label == +1);
            idx_neg = find(all_label == -1);

            data_pos = all_data(idx_pos, :);
            data_neg = all_data(idx_neg, :);

            n_pos = size(data_pos, 1);
            n_neg = size(data_neg, 1);

            % 各foldごとにインデックスを用意
            idx_p = 1:n_pos;
            idx_n = 1:n_neg;

            accuracy_list = zeros(cv, 1);  % 各foldの精度を格納するリスト

            % k-fold
            for i = 1:cv
                % 各foldにおいて、正例と負例を分割
                train_pos = data_pos(mod(idx_p, cv) ~= (i-1), :);
                eval_pos  = data_pos(mod(idx_p, cv) == (i-1), :);

                train_neg = data_neg(mod(idx_n, cv) ~= (i-1), :);
                eval_neg  = data_neg(mod(idx_n, cv) == (i-1), :);

                % 学習用データとラベル
                train_data = [train_pos; train_neg];
                train_label = [ones(size(train_pos, 1), 1); -ones(size(train_neg, 1), 1)];
                % 評価用データとラベル
                eval_data = [eval_pos; eval_neg];
                eval_label = [ones(size(eval_pos, 1), 1); -ones(size(eval_neg, 1), 1)];

                % SVM
                model_cv = fitcsvm(train_data, train_label, 'KernelFunction', kernelType, 'Standardize', true);
                Return = model_cv;

                % 予測ラベル
                predicted_label = predict(model_cv, eval_data);

                % 精度計算
                correct = sum(predicted_label == eval_label);
                ac = correct / length(eval_label);
                accuracy_list(i) = ac;

                fprintf(' Fold %d: accuracy = %.2f%%\n', i, ac * 100);
            end

            % 平均精度
            mean_ac = mean(accuracy_list);
            fprintf('Mean accuracy = %.2f%%\n', mean_ac * 100);

            % 最終モデルの学習
            model = fitcsvm(all_data, all_label, 'KernelFunction', kernelType, 'Standardize', true);
            fprintf('Final model trained\n');

            predicted_label = predict(model, all_data);

            % 精度計算
            correct = sum(predicted_label == all_label);
            ac = correct / length(all_label);
            fprintf('Final: accuracy = %.2f%%\n', ac * 100);
        end
    end
    
    methods(Static, Access=private)
        % 与えられたデータを写像する関数
        function data3 = map_chi2(data)
            % カイ2乗近似写像
            % data => n×d行列
            % data3 => n×(3d)行列

            data3 = repmat( sqrt(abs(data)) .* sign(data), [1,3]) .* ...
                    [0.8*ones(size(data)), ...
                     0.6*cos(0.6*log(abs(data)+eps)), ...
                     0.6*sin(0.6*log(abs(data)+eps))];
        end
    end
end