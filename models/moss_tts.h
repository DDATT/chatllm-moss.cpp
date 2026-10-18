#pragma once

#include "../src/models.h"
#include "../src/models_priv.h"
#include "qwen.h"

namespace chatllm::moss::codec
{
    enum
    {
        MAX_CODEC_STAGES = 16,
        MAX_QUANTIZERS   = 32,
    };

    struct StageConfig
    {
        int kind;       // 0 = Patch, 1 = Transformer
        int patch_size;
        int in_dim;
        int d_model;
        int n_heads;
        int n_layers;
        int d_ff;
        int out_dim;
        int context;
    };

    struct CodecConfig
    {
        int sample_rate;
        int downsample_rate;
        int num_quantizers;
        int codebook_size;
        int codebook_dim;
        int rvq_dim;
        int code_dim;
        int channels;
        int channel_interleave;
        float context_seconds;

        int n_enc_stages;
        int enc_kind[MAX_CODEC_STAGES];
        int enc_patch[MAX_CODEC_STAGES];
        int enc_in_dim[MAX_CODEC_STAGES];
        int enc_d_model[MAX_CODEC_STAGES];
        int enc_n_heads[MAX_CODEC_STAGES];
        int enc_n_layers[MAX_CODEC_STAGES];
        int enc_d_ff[MAX_CODEC_STAGES];
        int enc_out_dim[MAX_CODEC_STAGES];

        int n_dec_stages;
        int dec_kind[MAX_CODEC_STAGES];
        int dec_patch[MAX_CODEC_STAGES];
        int dec_in_dim[MAX_CODEC_STAGES];
        int dec_d_model[MAX_CODEC_STAGES];
        int dec_n_heads[MAX_CODEC_STAGES];
        int dec_n_layers[MAX_CODEC_STAGES];
        int dec_d_ff[MAX_CODEC_STAGES];
        int dec_out_dim[MAX_CODEC_STAGES];
    };

    class MOSSTransformerLayer : public Block
    {
    public:
        MOSSTransformerLayer(InitContext *ctx, int d_model, int n_heads, int d_ff);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *x, ggml::tensor *pos, ggml::tensor *mask);
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        const int d_model;
        const int n_heads;
        const int d_ff;
        LayerNorm norm1;
        Linear    qkv;
        Linear    out_proj;
        ggml::tensor *ls1;
        LayerNorm norm2;
        Linear    ffn_0;
        Linear    ffn_2;
        ggml::tensor *ls2;
    };

    class MOSSTransformerStage : public Block
    {
    public:
        MOSSTransformerStage(InitContext *ctx, const StageConfig &cfg);
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *x, ggml::tensor *pos, ggml::tensor *mask);
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        StageConfig cfg;
        std::unique_ptr<Linear> input_proj;
        std::vector<std::unique_ptr<MOSSTransformerLayer>> layers;
        std::unique_ptr<Linear> output_proj;
    };

    class MOSSResidualLFQ : public Block
    {
    public:
        MOSSResidualLFQ(InitContext *ctx, int n_quantizers, int codebook_size, int codebook_dim, int rvq_dim, int code_dim);
        ggml::tensor *quantize(ComputeContext *ctx, ggml::tensor *latent);
        ggml::tensor *dequantize(ComputeContext *ctx, ggml::tensor *codes, int k = -1);
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;
    public:
        const int n_quantizers;
        const int codebook_size;
        const int codebook_dim;
        const int rvq_dim;
        const int code_dim;
        Linear input_proj;
        Linear output_proj;
        std::vector<std::unique_ptr<Linear>> in_proj;
        std::vector<std::unique_ptr<Linear>> out_proj;
        std::vector<ggml::tensor *> codebooks;
    };

    class MOSSAudioTokenizer : public Block
    {
    public:
        struct TowerStage
        {
            StageConfig cfg;
            bool up = false;
            std::unique_ptr<MOSSTransformerStage> transformer;
        };

        MOSSAudioTokenizer(InitContext *ctx, const CodecConfig &cfg);
        int64_t get_param_num(bool effective_only) const override;
        void load(const std::string &path, TensorLoader *loader) override;

        bool encode(TensorGraphEvaluator &eval, const GenerationConfig &gen_config,
                    const std::vector<float> &pcm_samples, std::vector<int> &codes_out, int &n_frames_out);
        bool decode(TensorGraphEvaluator &eval, const GenerationConfig &gen_config,
                    const std::vector<int> &codes, int n_frames, int n_vq_used, std::vector<float> &pcm_samples);

    public:
        CodecConfig cfg;
        std::vector<TowerStage> encoder_stages;
        std::vector<TowerStage> decoder_stages;
        MOSSResidualLFQ quantizer;
    };
}

namespace chatllm::moss::tts
{
    struct Config : public qwen::v3::Config
    {
        // Global & local architecture flags
        int global_arch;            // 0 = Qwen3 (v1.5 / v1.0), 1 = GPT-2 + Interleaved RoPE (Nano)
        int local_arch;             // 0 = GPT-J/SiLU (v1.5), 1 = Qwen3 (v1.0), 2 = GPT-2/GELU (Nano)
        int local_hidden_size;
        int local_num_layers;
        int local_num_heads;
        int local_num_kv_heads;
        int local_head_dim;
        int local_ffn_hidden_size;
        float local_rope_theta;
        float local_norm_eps;

        // Multi-channel & adapter config
        int n_vq;                   // 12 (v1.5), 32 (v1.0), or 16 (Nano)
        int audio_vocab_size;       // 1024 (v1.5 / Nano) or 1025 (v1.0)
        int audio_embed_rows;       // 1025 (padded with 1 zero row when audio_pad_code == 1024)
        int additional_mlp_ffn_size;
        int bare_heads;             // 1 (v1.5 / Nano) or 0 (v1.0)
        int local_text_head_mode;   // 1 = binary [2, H] (v1.5), 0 = full text vocab (v1.0), 2 = Nano decision-then-16-codebooks
        int stereo;                 // 1 (v1.5 / Nano) or 0 (v1.0)
        int sample_rate;            // 48000 (v1.5 / Nano) or 24000 (v1.0)

        // Special token IDs
        int pad_token_id;
        int im_start_token_id;
        int im_end_token_id;
        int audio_start_token_id;
        int audio_end_token_id;
        int audio_user_slot_token_id;
        int audio_assistant_gen_slot_token_id;
        int audio_assistant_delay_slot_token_id;
        int audio_pad_code;

        // Embedded MOSS-Audio-Tokenizer config
        int has_codec;
        codec::CodecConfig codec_cfg;
    };

    class Tokenizer : public qwen::v3::Tokenizer
    {
    public:
        Tokenizer(const BaseConfig &cfg);
        size_t load(tokenizer::DataReader *buffer, int n_vocab) override;
        void encode(const std::string &text, std::vector<int> &ids) const override;
    public:
        bool is_nano = false;
        mutable std::string last_text;
        std::string instruction = "None";
        std::string language    = "None";
        std::string ref_audio   = "";
    };

    class LocalGeneration : public Block
    {
    public:
        LocalGeneration(const Config &config, const RuntimeConfig &runtime_config, size_t GRAPH_SIZE = 4096);
        void load(ModelLoader &loader);

        // Generate 1 + n_vq tokens (channel 0 = text, channels 1..n_vq = audio) given global hidden state
        bool generate_frame(const GenerationConfig &gen_config,
                            Block *text_embedding,
                            const float *global_hidden,
                            float text_temp, float text_top_p, int text_top_k,
                            float audio_temp, float audio_top_p, int audio_top_k, float audio_rep_penalty,
                            std::vector<std::vector<int>> &history,
                            std::vector<int> &next_frame);

    public:
        const Config config;
        TensorGraphEvaluator eval;
        InitContext _ctx;
        Sequential layers;
        std::unique_ptr<Block> final_norm;
        std::vector<std::unique_ptr<Embedding>> audio_embeddings;  // [0..n_vq-1] for audio channels 1..n_vq
        std::unique_ptr<Block> in_mlp;                             // v1.0 shared speech->local MLP
        std::vector<std::unique_ptr<Block>> out_mlps;              // v1.0 per-channel local->speech MLPs
        std::vector<std::unique_ptr<RMSNorm>> head_norms;          // v1.0 pre-head RMSNorms
        std::vector<std::unique_ptr<Linear>> lm_heads;             // [0] = text (or nullptr in v1.5), [1..n_vq] = audio
        std::unique_ptr<Linear> local_text_head;                   // v1.5 binary [2, H] head
        bool loaded = false;
    };

    class ConditionalGeneration : public BaseModelForConditionalGeneration
    {
    public:
        typedef BaseModelForConditionalGeneration Base;
        typedef HeterogeneousModel ModelClass;
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = (ModelType)MODEL_TYPE_MOSS_TTS);
        void load(ModelLoader &loader) override;
        void set_additional_args(const std::map<std::string, std::string> &args) override;
        void speech_synthesis(const GenerationConfig &gen_config, const std::vector<int> &input_ids,
                              std::vector<int16_t> &audio, int &sample_rate, int &channels) override;

    protected:
        bool run_global_step(const GenerationConfig &gen_config,
                             const std::vector<int> &row_major_ids,
                             int n_rows,
                             std::vector<float> &last_hidden_out);

    public:
        const Config config;
        LocalGeneration local_gen;
        TensorGraphEvaluator codec_eval;
        InitContext codec_ctx;
        std::unique_ptr<codec::MOSSAudioTokenizer> audio_tokenizer;
        bool codec_loaded = false;

        // Sampling overrides via --additional_args
        float text_temperature = 1.5f;
        float text_top_p = 1.0f;
        int   text_top_k = 50;
        float audio_temperature = 1.7f;
        float audio_top_p = 0.8f;
        int   audio_top_k = 25;
        float audio_repetition_penalty = 1.0f;
        bool  greedy = false;
    };
}
