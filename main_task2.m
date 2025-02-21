function main_task2()
    % asakusaで25枚
    run_task2("asakusa", 25);
    % asakusaで50枚
    run_task2("asakusa", 50);
    % kyotoで25枚
    run_task2("kyoto", 25);
    % kyotoで50枚
    run_task2("kyoto", 50);
end

% region: asakusaかkyoto
% numTrain: 何枚で訓練するか（25 または 50）
function run_task2(region, numTrain)
    fprintf("\n=======課題２：「%s」で%d枚=======\n", region, numTrain);

    % 訓練用画像リストとラベルの取得
    train_dir = sprintf("./%s", region);
    [img_list, img_label] = utils.create_img_list_with_label_new(train_dir, numTrain, "./bgimg", 1000);

    % DDCNNエンコード
    net = vgg16;
    encoded_v = encode.DCNN_mlt(img_list, net, 'fc7');

    % SVMを学習
    model = classification.svm_kfold_new(encoded_v, img_label, "VGG16_fc7", 5, "linear");

    % テスト画像のリスト取得 (テストディレクトリは region+_test、asakusaだったらasakusa_test)
    test_dir = sprintf("%s_test", region);
    test_files = dir(fullfile(test_dir, '*.jpg'));
    test_img_list = cell(numel(test_files), 1);
    for i = 1:numel(test_files)
        test_img_list{i} = fullfile(test_files(i).folder, test_files(i).name);
    end

    % テスト画像のDCNN特徴抽出
    encoded_v_test = encode.DCNN_mlt(test_img_list, net, 'fc7');

    % 予測
    [pred_label, score] = predict(model, encoded_v_test);

    % SVMの2番目の出力スコアを降順にソート
    [sorted_score, sorted_idx] = sort(score(:,2), 'descend');

    % ソートされた順に画像ファイル名とスコアを表示
    for i = 1:numel(sorted_idx)
        [~, fname, fext] = fileparts(test_img_list{sorted_idx(i)});
        fprintf('%s %f\n', [fname, fext], sorted_score(i));
    end
end