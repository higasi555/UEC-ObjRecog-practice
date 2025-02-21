classdef encode
    methods(Static)
        function Return = ColorHist64(img_list)
            n = length(img_list);
            database = zeros(n, 64);
            for i = 1 : n
                img = imread(img_list{i});
                R = img(:,:,1);
                G = img(:,:,2);
                B = img(:,:,3);
                idx64 = uint8(floor(double(R) / 64) * 16 + floor(double(G) / 64) * 4 + floor(double(B) / 64));
                h = histcounts(idx64(:), 0 : 64);
                database(i,:) = h / sum(h);
            end
            Return = database;
        end
                
        function Return = BoF(img_list, path2codebook)
            % 画像数n
            n = length(img_list);
        
            % codebookの読み込み
            load(path2codebook,'codebook_v');
            k = size(codebook_v,1);
        
            bof = zeros(n, k);
            for j = 1:n
                grayI = rgb2gray(imread(img_list{j}));
        
                % SURF特徴点検出
                % 以下どちらか選ぶ
                pts = encode.createRandomPoints(grayI, 1000); % Dense
%                pts = detectSURFFeatures(grayI); % Sparse
        
                % 強い特徴点を1000個選択
                pts = pts.selectStrongest(1000);
        
                % 特徴量を抽出
                [f, ~] = extractFeatures(grayI, pts);
        
                % BoFベクトル作成
                if ~isempty(f)
                    dist = utils.calc_dist_mat(double(f), double(codebook_v));
                    [~, idx_min] = min(dist, [], 2);  % 各特徴点に最も近いベクトルに投票
                    for m = 1:length(idx_min)
                        bof(j, idx_min(m)) = bof(j, idx_min(m)) + 1;
                    end
                end
            end
        
            % 正規化
            row_sum = sum(bof,2);
            row_sum(row_sum==0) = 1;
            bof = bof ./ row_sum;
        
            Return = bof;
        end
        
        function PT = createRandomPoints(I, num)
            [sy, sx] = size(I);
            sz = [sx, sy];
            for i=1:num
                s=0;
                while s<1.6
                    s=randn()*3+3;
                end
                p = ceil((sz-ceil(s)*2).*rand(1,2)+ceil(s));
                if i==1
                    PT = SURFPoints(p,'Scale',s);
                else
                    PT = [PT; SURFPoints(p,'Scale',s)];
                end
            end
        end
    
        function Return = DCNN(img_list, net)
            inputSize = net.Layers(1).InputSize(1:2);
            N = length(img_list);
            
            % N×4096のfc7
            fc7 = zeros(N, 4096, 'single');
            
            for i = 1:N
                img = imread(img_list{i});
                reimg = imresize(img, inputSize);
                
                % activationsを利用して中間特徴量を取り出します．
                % 4096次元の'fc7'から特徴抽出します．
                dcnnf = activations(net,reimg,'fc7');  
                
                % squeeze関数で，ベクトル化します．
                dcnnf = squeeze(dcnnf);
                
                % L2ノルムで割って，L2正規化．
                % 最終的な dcnnf を画像特徴量として利用します．
                dcnnf = dcnnf/norm(dcnnf);
                
                fc7(i,:) = dcnnf;         
            end

            Return = fc7;
        end

        function Return = DCNN_mlt(img_list, net, layerName)
            inputSize = net.Layers(1).InputSize(1:2);
            N = length(img_list);

            % 出力次元を切り替え
            switch layerName
                case 'fc6'
                    outdim = 4096;
                case 'fc7'
                    outdim = 4096;
                case 'fc8'
                    outdim = 1000;
                otherwise
                    error('Unsupported layerName: %s', layerName);
            end

            feats = zeros(N, outdim, 'single');

            for i = 1:N
                % デバッグ用
                % fprintf("reading img %d", i);
                img = imread(img_list{i});
                reimg = imresize(img, inputSize);

                dcnnf = activations(net, reimg, layerName);
                dcnnf = squeeze(dcnnf);
                dcnnf = dcnnf / norm(dcnnf);

                feats(i,:) = dcnnf;
            end

            Return = feats;
        end
    end
end

