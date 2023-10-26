#!/usr/bin/env bash

files=(
    resources/outputs/bloom-1b1/asset-test_asset-valid_p2_random_fs3_nr1_s489
    resources/outputs/bloom-1b1/asset-test_asset-valid_p2_random_fs3_nr1_s723
    resources/outputs/bloom-1b1/med-easi-test_med-easi-validation_p0_random_fs3_nr1_s287
    resources/outputs/llama-13b/asset-test_asset-valid_p0_random_fs3_nr1_s723
    resources/outputs/llama-13b/med-easi-test_med-easi-validation_p2_random_fs3_nr1_s489
    resources/outputs/llama-13b/med-easi-test_med-easi-validation_p2_random_fs3_nr1_s723
    resources/outputs/llama-13b/news-manual-all-test_news-manual-all-val_p0_random_fs3_nr1_s287
    resources/outputs/llama-13b/news-manual-all-test_news-manual-all-val_p0_random_fs3_nr1_s489
    resources/outputs/llama-13b/news-manual-all-test_news-manual-all-val_p1_random_fs3_nr1_s489
    resources/outputs/llama-13b/news-manual-all-test_news-manual-all-val_p1_random_fs3_nr1_s723
    resources/outputs/llama-13b/news-manual-all-test_news-manual-all-val_p2_random_fs3_nr1_s489
    resources/outputs/llama-13b/news-manual-all-test_news-manual-all-val_p2_random_fs3_nr1_s723
    resources/outputs/llama-30b/asset-test_asset-valid_p0_random_fs3_nr1_s287
    resources/outputs/llama-30b/asset-test_asset-valid_p0_random_fs3_nr1_s489
    resources/outputs/llama-30b/asset-test_asset-valid_p1_random_fs3_nr1_s287
    resources/outputs/llama-30b/asset-test_asset-valid_p1_random_fs3_nr1_s489
    resources/outputs/llama-30b/asset-test_asset-valid_p2_random_fs3_nr1_s287
    resources/outputs/llama-30b/asset-test_asset-valid_p2_random_fs3_nr1_s489
    resources/outputs/llama-30b/asset-test_asset-valid_p2_random_fs3_nr1_s723
    resources/outputs/llama-30b/med-easi-test_med-easi-validation_p0_random_fs3_nr1_s489
    resources/outputs/llama-30b/med-easi-test_med-easi-validation_p1_random_fs3_nr1_s489
    resources/outputs/llama-30b/med-easi-test_med-easi-validation_p1_random_fs3_nr1_s723
    resources/outputs/llama-30b/med-easi-test_med-easi-validation_p2_random_fs3_nr1_s489
    resources/outputs/llama-30b/news-manual-all-test_news-manual-all-val_p0_random_fs3_nr1_s287
    resources/outputs/llama-30b/news-manual-all-test_news-manual-all-val_p0_random_fs3_nr1_s489
    resources/outputs/llama-30b/news-manual-all-test_news-manual-all-val_p2_random_fs3_nr1_s287
    resources/outputs/llama-65b/asset-test_asset-valid_p0_random_fs3_nr1_s287
    resources/outputs/llama-65b/asset-test_asset-valid_p0_random_fs3_nr1_s489
    resources/outputs/llama-65b/asset-test_asset-valid_p1_random_fs3_nr1_s489
    resources/outputs/llama-65b/med-easi-test_med-easi-validation_p0_random_fs3_nr1_s489
    resources/outputs/llama-65b/med-easi-test_med-easi-validation_p0_random_fs3_nr1_s723
    resources/outputs/llama-65b/med-easi-test_med-easi-validation_p1_random_fs3_nr1_s287
    resources/outputs/llama-65b/med-easi-test_med-easi-validation_p2_random_fs3_nr1_s287
    resources/outputs/llama-65b/med-easi-test_med-easi-validation_p2_random_fs3_nr1_s489
    resources/outputs/llama-65b/news-manual-all-test_news-manual-all-val_p0_random_fs3_nr1_s489
    resources/outputs/llama-65b/news-manual-all-test_news-manual-all-val_p0_random_fs3_nr1_s723
    resources/outputs/llama-65b/news-manual-all-test_news-manual-all-val_p1_random_fs3_nr1_s723
    resources/outputs/llama-65b/news-manual-all-test_news-manual-all-val_p2_random_fs3_nr1_s287
    resources/outputs/llama-7b/asset-test_asset-valid_p0_random_fs3_nr1_s287
    resources/outputs/llama-7b/asset-test_asset-valid_p2_random_fs3_nr1_s489
    resources/outputs/llama-7b/med-easi-test_med-easi-validation_p0_random_fs3_nr1_s287
    resources/outputs/llama-7b/med-easi-test_med-easi-validation_p0_random_fs3_nr1_s489
    resources/outputs/llama-7b/med-easi-test_med-easi-validation_p0_random_fs3_nr1_s723
    resources/outputs/llama-7b/med-easi-test_med-easi-validation_p1_random_fs3_nr1_s489
    resources/outputs/llama-7b/med-easi-test_med-easi-validation_p1_random_fs3_nr1_s723
    resources/outputs/llama-7b/med-easi-test_med-easi-validation_p2_random_fs3_nr1_s489
    resources/outputs/llama-7b/med-easi-test_med-easi-validation_p2_random_fs3_nr1_s723
    resources/outputs/llama-7b/news-manual-all-test_news-manual-all-val_p0_random_fs3_nr1_s723
    resources/outputs/llama-7b/news-manual-all-test_news-manual-all-val_p1_random_fs3_nr1_s489
    resources/outputs/llama-7b/news-manual-all-test_news-manual-all-val_p2_random_fs3_nr1_s287
    resources/outputs/llama-7b/news-manual-all-test_news-manual-all-val_p2_random_fs3_nr1_s723
    resources/outputs/ul2/asset-test_asset-valid_p0_random_fs3_nr1_s723
    resources/outputs/ul2/asset-test_asset-valid_p1_random_fs3_nr1_s489
    resources/outputs/ul2/asset-test_asset-valid_p1_random_fs3_nr1_s723
    resources/outputs/ul2/med-easi-test_med-easi-validation_p0_random_fs3_nr1_s287
    resources/outputs/ul2/med-easi-test_med-easi-validation_p0_random_fs3_nr1_s489
    resources/outputs/ul2/med-easi-test_med-easi-validation_p0_random_fs3_nr1_s723
    resources/outputs/ul2/med-easi-test_med-easi-validation_p2_random_fs3_nr1_s287
    resources/outputs/ul2/med-easi-test_med-easi-validation_p2_random_fs3_nr1_s489
)

for file in "${files[@]}"; do
    infile="$file.jsonl"
    outfile="$file.eval"
    echo "rescoring $infile"
    echo "writing to $outfile"
    
    python -m evaluation.simplification_evaluation $infile \
        --out_file $outfile \
        --use_cuda \
        --lens_model_path "resources/LENS/checkpoints/epoch=5-step=6102.ckpt"

done