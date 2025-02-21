classdef codebook
    methods(Static)
        function Return = gen_codebook(img_list, path2codebook)
            % ランダム点抽出でSURF特徴収集
            numPointsPerImage = 1000;
            all_features = []; % 全てのSURF特徴量を格納する行列
            % 特徴抽出
            for i = 1 : length(img_list)
                grayI = rgb2gray(imread(img_list{i}));
                % 以下どちらか選ぶ
                pts = encode.createRandomPoints(grayI, 3000); % Dense
%                pts = detectSURFFeatures(grayI); % Sparse

                % 強い特徴点を300個選択
                pts = pts.selectStrongest(numPointsPerImage);
                
                [features, ~] = extractFeatures(grayI, pts);
                % featuresはM×64の実数行列
                all_features = [all_features; features];
            end
        
            % 5万件に制限
            if size(all_features,1) > 50000
                sel = randperm(size(all_features,1), 50000);
                all_features = all_features(sel,:);
            end
        
            % k=500でk-means
            k = 500;
            [~, codebook_v] = kmeans(double(all_features), k, 'MaxIter',1000, 'Display','final');
            save(path2codebook,'codebook_v');
            disp('Codebook saved');
        end
    end
end
