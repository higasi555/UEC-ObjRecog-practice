classdef utils
    methods(Static)
        function [img_list, labels] = create_img_list_with_label_new(pos_dir, pos_num, neg_dir, neg_num)
            % ポジティブ画像の取得
            pos_files = dir(fullfile(pos_dir, '*.jpg'));
            [~, order] = sort({pos_files.name});
            pos_files = pos_files(order);
            pos_img_list = cell(pos_num, 1);
            for i = 1:pos_num
                pos_img_list{i} = fullfile(pos_files(i).folder, pos_files(i).name);
            end

            % ネガティブ画像の取得
            neg_files = dir(fullfile(neg_dir, '*.jpg'));
            [~, order] = sort({neg_files.name});
            neg_files = neg_files(order);
            neg_img_list = cell(neg_num, 1);
            for i = 1:neg_num
                neg_img_list{i} = fullfile(neg_files(i).folder, neg_files(i).name);
            end

            % 合計画像リストとラベルの作成
            img_list = [pos_img_list; neg_img_list];
            labels = [ones(pos_num, 1); -ones(neg_num, 1)];
        end

        % 距離を求める
        function Return = calc_dist_mat(A, B)
            % A: NxD, B: kxD
            A_sq = sum(A.^2,2);    % Nx1
            B_sq = sum(B.^2,2)';   % 1xk
            Return = bsxfun(@plus, A_sq, B_sq) - 2*(A*B');
        end

        function save_classification_results(imageList, trueLabels, predictedLabels, methodName, baseDir)
            % 各カテゴリ用のディレクトリ作成
            TP_dir = fullfile(baseDir, methodName, 'TP');
            FP_dir = fullfile(baseDir, methodName, 'FP');
            TN_dir = fullfile(baseDir, methodName, 'TN');
            FN_dir = fullfile(baseDir, methodName, 'FN');

            if ~exist(TP_dir, 'dir'), mkdir(TP_dir); end
            if ~exist(FP_dir, 'dir'), mkdir(FP_dir); end
            if ~exist(TN_dir, 'dir'), mkdir(TN_dir); end
            if ~exist(FN_dir, 'dir'), mkdir(FN_dir); end

            % すべての画像について判定し、該当ディレクトリへコピー
            for i = 1:length(imageList)
                % 画像ファイル名からファイル名と拡張子を抽出
                [~, fname, fext] = fileparts(imageList{i});
                destFile = [fname, fext];

                % 真のラベルと予測ラベルによる4分類
                if trueLabels(i) == 1 && predictedLabels(i) == 1
                    % True Positive
                    copyfile(imageList{i}, fullfile(TP_dir, destFile));
                elseif trueLabels(i) == 1 && predictedLabels(i) == -1
                    % False Negative
                    copyfile(imageList{i}, fullfile(FN_dir, destFile));
                elseif trueLabels(i) == -1 && predictedLabels(i) == 1
                    % False Positive
                    copyfile(imageList{i}, fullfile(FP_dir, destFile));
                elseif trueLabels(i) == -1 && predictedLabels(i) == -1
                    % True Negative
                    copyfile(imageList{i}, fullfile(TN_dir, destFile));
                else
                    % 万が一、その他のラベルがある場合
                    warning('Unexpected label combination at index %d', i);
                end
            end

            fprintf('Method %s: TP, FP, TN, FN の結果を保存完了\n', methodName);
        end

    end
end
