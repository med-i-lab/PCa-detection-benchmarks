
# for FOLD in {0..9}
# do
#     python train_ensemble.py n_folds=10 fold=${FOLD} use_calibration=false exp_name=ensemble_fold_nocal${FOLD} +num_seeds=10
# done

python test_ensemble.py --exp_dir /h/pwilson/projects/TRUSnet/projects/IJCARS_2023/outputs/ensemble_fold_nocal0/2023-05-11_22-28-39 --fold 0 --n_folds 10 --output_suffix _no_tc
python test_ensemble.py --exp_dir /h/pwilson/projects/TRUSnet/projects/IJCARS_2023/outputs/ensemble_fold_nocal1/2023-05-11_23-32-07 --fold 1 --n_folds 10 --output_suffix _no_tc
python test_ensemble.py --exp_dir /h/pwilson/projects/TRUSnet/projects/IJCARS_2023/outputs/ensemble_fold_nocal2/2023-05-12_00-38-46 --fold 2 --n_folds 10 --output_suffix _no_tc
python test_ensemble.py --exp_dir /h/pwilson/projects/TRUSnet/projects/IJCARS_2023/outputs/ensemble_fold_nocal3/2023-05-12_01-49-37 --fold 3 --n_folds 10 --output_suffix _no_tc
python test_ensemble.py --exp_dir /h/pwilson/projects/TRUSnet/projects/IJCARS_2023/outputs/ensemble_fold_nocal4/2023-05-12_03-01-50 --fold 4 --n_folds 10 --output_suffix _no_tc
python test_ensemble.py --exp_dir /h/pwilson/projects/TRUSnet/projects/IJCARS_2023/outputs/ensemble_fold_nocal5/2023-05-12_04-12-35 --fold 5 --n_folds 10 --output_suffix _no_tc
python test_ensemble.py --exp_dir /h/pwilson/projects/TRUSnet/projects/IJCARS_2023/outputs/ensemble_fold_nocal6/2023-05-12_05-25-21 --fold 6 --n_folds 10 --output_suffix _no_tc
python test_ensemble.py --exp_dir /h/pwilson/projects/TRUSnet/projects/IJCARS_2023/outputs/ensemble_fold_nocal7/2023-05-12_06-39-15 --fold 7 --n_folds 10 --output_suffix _no_tc
python test_ensemble.py --exp_dir /h/pwilson/projects/TRUSnet/projects/IJCARS_2023/outputs/ensemble_fold_nocal8/2023-05-12_07-50-19 --fold 8 --n_folds 10 --output_suffix _no_tc
python test_ensemble.py --exp_dir /h/pwilson/projects/TRUSnet/projects/IJCARS_2023/outputs/ensemble_fold_nocal9/2023-05-12_09-00-29 --fold 9 --n_folds 10 --output_suffix _no_tc

python test_ensemble.py --exp_dir /h/pwilson/projects/TRUSnet/projects/IJCARS_2023/outputs/ensemble_fold_nocal0/2023-05-11_22-28-39 --fold 0 --n_folds 10 --output_suffix _late_tc --late_tc
python test_ensemble.py --exp_dir /h/pwilson/projects/TRUSnet/projects/IJCARS_2023/outputs/ensemble_fold_nocal1/2023-05-11_23-32-07 --fold 1 --n_folds 10 --output_suffix _late_tc --late_tc
python test_ensemble.py --exp_dir /h/pwilson/projects/TRUSnet/projects/IJCARS_2023/outputs/ensemble_fold_nocal2/2023-05-12_00-38-46 --fold 2 --n_folds 10 --output_suffix _late_tc --late_tc
python test_ensemble.py --exp_dir /h/pwilson/projects/TRUSnet/projects/IJCARS_2023/outputs/ensemble_fold_nocal3/2023-05-12_01-49-37 --fold 3 --n_folds 10 --output_suffix _late_tc --late_tc
python test_ensemble.py --exp_dir /h/pwilson/projects/TRUSnet/projects/IJCARS_2023/outputs/ensemble_fold_nocal4/2023-05-12_03-01-50 --fold 4 --n_folds 10 --output_suffix _late_tc --late_tc
python test_ensemble.py --exp_dir /h/pwilson/projects/TRUSnet/projects/IJCARS_2023/outputs/ensemble_fold_nocal5/2023-05-12_04-12-35 --fold 5 --n_folds 10 --output_suffix _late_tc --late_tc
python test_ensemble.py --exp_dir /h/pwilson/projects/TRUSnet/projects/IJCARS_2023/outputs/ensemble_fold_nocal6/2023-05-12_05-25-21 --fold 6 --n_folds 10 --output_suffix _late_tc --late_tc
python test_ensemble.py --exp_dir /h/pwilson/projects/TRUSnet/projects/IJCARS_2023/outputs/ensemble_fold_nocal7/2023-05-12_06-39-15 --fold 7 --n_folds 10 --output_suffix _late_tc --late_tc
python test_ensemble.py --exp_dir /h/pwilson/projects/TRUSnet/projects/IJCARS_2023/outputs/ensemble_fold_nocal8/2023-05-12_07-50-19 --fold 8 --n_folds 10 --output_suffix _late_tc --late_tc
python test_ensemble.py --exp_dir /h/pwilson/projects/TRUSnet/projects/IJCARS_2023/outputs/ensemble_fold_nocal9/2023-05-12_09-00-29 --fold 9 --n_folds 10 --output_suffix _late_tc --late_tc