function main_task1_2()
    fprintf("\n=======課題１：うどんとそば=======\n");
    [img_list, img_label] = utils.create_img_list_with_label_new("./soba", 100, "./udon", 100);

    codebook.gen_codebook(img_list, "codebook.mat");

    % 各種encode
    encoded_v_Hist64 = encode.ColorHist64(img_list);
    encoded_v_BoF = encode.BoF(img_list, "codebook.mat");

    net = vgg16;
    encoded_v_vgg16_DCNN_fc7_train = encode.DCNN_mlt(img_list, net, 'fc7');

    % 各種分類
    model = classification.svm_kfold_new_model5(encoded_v_Hist64, img_label, "Hist64", 5, "linear");
    [pred_label, score] = predict(model, encoded_v_Hist64);
    utils.save_classification_results(img_list, img_label, pred_label, "Hist64", "./Result_1-2");

    model = classification.svm_kfold_new_model5(encoded_v_BoF, img_label, "BoF", 5, "rbf");
    [pred_label, score] = predict(model, encoded_v_BoF);
    utils.save_classification_results(img_list, img_label, pred_label, "BoF", "./Result_1-2");

    model = classification.svm_kfold_new_model5(encoded_v_vgg16_DCNN_fc7_train, img_label, "VGG16_fc7", 5, "linear");
    [pred_label, score] = predict(model, encoded_v_vgg16_DCNN_fc7_train);
    utils.save_classification_results(img_list, img_label, pred_label, "VGG16_fc7", "./Result_1-2");

end
