clc; clear; close all;

base_folder = '.\0421\figure1\2D_Aus_org_1e-4\weak_focal_clean_alpha_0.25_gamma_2\result_all_percent_0.8_winsize_8820_winover_0.8';
final_CM = csvread([base_folder, '\confusion_matrix.csv']);

% base_folder = '.\plot\';
% final_CM = csvread([base_folder, '\best_cm.csv']);

plotConfMat_Aus(final_CM)

export_fig('best_CM_Aus.pdf')
% saveas(gcf,'point_weighted_F1_score.png')





