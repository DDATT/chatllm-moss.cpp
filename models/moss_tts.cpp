#include "moss_tts.h"
#include "../src/audio_process.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>

namespace chatllm::moss::codec
{
    static ggml::tensor *patch_down(ComputeContext *ctx, ggml::tensor *x, int p)
    {
        const int64_t T = ggml::get_dim(x, 0);
        const int64_t D = ggml::get_dim(x, 1);
        ggml::tensor *v = ggml::reshape_3d(ctx, x, p, T / p, D);
        v = ggml::cont(ctx, ggml::permute(ctx, v, 1, 0, 2, 3));
        return ggml::reshape_2d(ctx, v, T / p, D * p);
    }

    static ggml::tensor *patch_up(ComputeContext *ctx, ggml::tensor *x, int p)
    {
        const int64_t L  = ggml::get_dim(x, 0);
        const int64_t Dp = ggml::get_dim(x, 1);
        const int64_t D  = Dp / p;
        ggml::tensor *v = ggml::reshape_3d(ctx, x, L, p, D);
        v = ggml::cont(ctx, ggml::permute(ctx, v, 1, 0, 2, 3));
        return ggml::reshape_2d(ctx, v, L * p, D);
    }

    static void build_window_mask(std::vector<float> &dst, int T, int context)
    {
        dst.resize((size_t)T * (size_t)T);
        const float ninf = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < T; ++i)
        {
            for (int j = 0; j < T; ++j)
            {
                const int delta = i - j;
                bool allowed;
                if (context <= 0 || context >= T)
                    allowed = (delta >= 0);
                else
                    allowed = (delta >= 0 && delta < context);
                dst[(size_t)i * T + j] = allowed ? 0.0f : ninf;
            }
        }
    }

    MOSSTransformerLayer::MOSSTransformerLayer(InitContext *ctx, int d_model, int n_heads, int d_ff)
        : d_model(d_model),
          n_heads(n_heads),
          d_ff(d_ff),
          norm1(ctx, d_model, true, 1e-5f),
          qkv(ctx, d_model, 3 * d_model, false),
          out_proj(ctx, d_model, d_model, false),
          ls1(ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_F32, d_model)),
          norm2(ctx, d_model, true, 1e-5f),
          ffn_0(ctx, d_model, d_ff, false),
          ffn_2(ctx, d_ff, d_model, false),
          ls2(ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_F32, d_model))
    {
    }

    ggml::tensor *MOSSTransformerLayer::forward(ComputeContext *ctx, ggml::tensor *x, ggml::tensor *pos, ggml::tensor *mask)
    {
        const int64_t Tt = ggml::get_dim(x, 1);
        const int64_t D  = d_model;
        const int64_t H  = n_heads;
        const int64_t hd = D / H;

        ggml::tensor *h = norm1.forward(ctx, x);
        ggml::tensor *qkv_out = qkv.forward(ctx, h);

        ggml::tensor *q = ggml::view_2d(ctx, qkv_out, D, Tt, qkv_out->nb[1], 0);
        ggml::tensor *k = ggml::view_2d(ctx, qkv_out, D, Tt, qkv_out->nb[1], D * ggml::element_size(qkv_out));
        ggml::tensor *v = ggml::view_2d(ctx, qkv_out, D, Tt, qkv_out->nb[1], 2 * D * ggml::element_size(qkv_out));

        q = ggml::reshape_3d(ctx, ggml::cont(ctx, q), hd, H, Tt);
        k = ggml::reshape_3d(ctx, ggml::cont(ctx, k), hd, H, Tt);
        v = ggml::reshape_3d(ctx, ggml::cont(ctx, v), hd, H, Tt);

        q = ggml::rope_ext(ctx, q, pos, nullptr, (int)hd, RoPEMode::Interleaved, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml::rope_ext(ctx, k, pos, nullptr, (int)hd, RoPEMode::Interleaved, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        q = ggml::cont(ctx, ggml::permute(ctx, q, 0, 2, 1, 3));
        k = ggml::cont(ctx, ggml::permute(ctx, k, 0, 2, 1, 3));
        v = ggml::cont(ctx, ggml::permute(ctx, v, 0, 2, 1, 3));

        ggml::tensor *scores = ggml::mul_mat(ctx, k, q);
        const float scale = 1.0f / std::sqrt((float)hd);
        if (mask)
        {
            scores = ggml::soft_max_ext(ctx, scores, mask, scale, 0.0f);
        }
        else
        {
            scores = ggml::scale(ctx, scores, scale);
            scores = ggml::diag_mask_inf(ctx, scores, 0);
            scores = ggml::soft_max(ctx, scores);
        }

        ggml::tensor *vt = ggml::cont(ctx, ggml::transpose(ctx, v));
        ggml::tensor *o  = ggml::mul_mat(ctx, vt, scores);
        o = ggml::cont(ctx, ggml::permute(ctx, o, 0, 2, 1, 3));
        o = ggml::reshape_2d(ctx, o, D, Tt);
        ggml::tensor *a = out_proj.forward(ctx, o);

        x = ggml::add(ctx, x, ggml::mul(ctx, a, ls1));

        h = norm2.forward(ctx, x);
        h = ffn_0.forward(ctx, h);
        ggml::tensor *gelu_h = ggml_gelu_erf(ctx->get_ctx(), h);
        ctx->cb_op_tensor(gelu_h);
        h = ffn_2.forward(ctx, gelu_h);

        x = ggml::add(ctx, x, ggml::mul(ctx, h, ls2));
        return x;
    }

    int64_t MOSSTransformerLayer::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += norm1.get_param_num(effective_only);
        r += qkv.get_param_num(effective_only);
        r += out_proj.get_param_num(effective_only);
        r += ggml::nelements(ls1);
        r += norm2.get_param_num(effective_only);
        r += ffn_0.get_param_num(effective_only);
        r += ffn_2.get_param_num(effective_only);
        r += ggml::nelements(ls2);
        return r;
    }

    void MOSSTransformerLayer::load(const std::string &path, TensorLoader *loader)
    {
        norm1.load(path + "norm1.", loader);
        norm2.load(path + "norm2.", loader);
        if (loader->has_tensor(path + "self_attn.in_proj.weight"))
            qkv.load(path + "self_attn.in_proj.", loader);
        else
            qkv.load(path + "self_attn.in_projs.0.", loader);

        if (loader->has_tensor(path + "self_attn.out_proj.weight"))
            out_proj.load(path + "self_attn.out_proj.", loader);
        else
            out_proj.load(path + "self_attn.out_projs.0.", loader);

        if (loader->has_tensor(path + "ffn.0.weight"))
            ffn_0.load(path + "ffn.0.", loader);
        else
            ffn_0.load(path + "linear1.", loader);

        if (loader->has_tensor(path + "ffn.2.weight"))
            ffn_2.load(path + "ffn.2.", loader);
        else
            ffn_2.load(path + "linear2.", loader);

        loader->read_tensor(path + "layer_scale_1.scale", ls1);
        loader->read_tensor(path + "layer_scale_2.scale", ls2);
    }

    MOSSTransformerStage::MOSSTransformerStage(InitContext *ctx, const StageConfig &cfg)
        : cfg(cfg)
    {
        if (cfg.in_dim != cfg.d_model)
            input_proj.reset(new Linear(ctx, cfg.in_dim, cfg.d_model, false));
        for (int i = 0; i < cfg.n_layers; ++i)
            layers.emplace_back(new MOSSTransformerLayer(ctx, cfg.d_model, cfg.n_heads, cfg.d_ff));
        if (cfg.out_dim != cfg.d_model)
            output_proj.reset(new Linear(ctx, cfg.d_model, cfg.out_dim, false));
    }

    ggml::tensor *MOSSTransformerStage::forward(ComputeContext *ctx, ggml::tensor *x, ggml::tensor *pos, ggml::tensor *mask)
    {
        if (input_proj)
            x = input_proj->forward(ctx, x);
        for (auto &layer : layers)
            x = layer->forward(ctx, x, pos, mask);
        if (output_proj)
            x = output_proj->forward(ctx, x);
        return x;
    }

    int64_t MOSSTransformerStage::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        if (input_proj) r += input_proj->get_param_num(effective_only);
        for (const auto &l : layers) r += l->get_param_num(effective_only);
        if (output_proj) r += output_proj->get_param_num(effective_only);
        return r;
    }

    void MOSSTransformerStage::load(const std::string &path, TensorLoader *loader)
    {
        if (input_proj && loader->has_tensor(path + "input_proj.weight"))
            input_proj->load(path + "input_proj.", loader);
        for (size_t i = 0; i < layers.size(); ++i)
            layers[i]->load(path + "transformer.layers." + std::to_string(i) + ".", loader);
        if (output_proj && loader->has_tensor(path + "output_proj.weight"))
            output_proj->load(path + "output_proj.", loader);
    }

    MOSSResidualLFQ::MOSSResidualLFQ(InitContext *ctx, int n_quantizers, int codebook_size, int codebook_dim, int rvq_dim, int code_dim)
        : n_quantizers(n_quantizers),
          codebook_size(codebook_size),
          codebook_dim(codebook_dim),
          rvq_dim(rvq_dim),
          code_dim(code_dim),
          input_proj(ctx, code_dim, rvq_dim, true),
          output_proj(ctx, rvq_dim, code_dim, true)
    {
        for (int i = 0; i < n_quantizers; ++i)
        {
            in_proj.emplace_back(new Linear(ctx, rvq_dim, codebook_dim, true));
            out_proj.emplace_back(new Linear(ctx, codebook_dim, rvq_dim, true));
            codebooks.push_back(ggml::new_tensor_2d(ctx, ggml::type::GGML_TYPE_F32, codebook_dim, codebook_size));
        }
    }

    ggml::tensor *MOSSResidualLFQ::quantize(ComputeContext *ctx, ggml::tensor *latent)
    {
        ggml::tensor *residual = input_proj.forward(ctx, latent);
        std::vector<ggml::tensor *> cols;
        for (int i = 0; i < n_quantizers; ++i)
        {
            ggml::tensor *ze  = in_proj[i]->forward(ctx, residual);
            ggml::tensor *e   = ggml::norm_p2(ctx, ze, 1e-12f);
            ggml::tensor *c   = ggml::norm_p2(ctx, codebooks[i], 1e-12f);
            ggml::tensor *sim = ggml::mul_mat(ctx, c, e);
            ggml::tensor *idx = ggml_argmax(ctx->get_ctx(), sim);
            ctx->cb_op_tensor(idx);
            cols.push_back(ggml::reshape_2d(ctx, idx, 1, ggml::get_dim(idx, 0)));
            ggml::tensor *sel = ggml::get_rows(ctx, codebooks[i], idx);
            ggml::tensor *zq  = out_proj[i]->forward(ctx, sel);
            residual = ggml::sub(ctx, residual, zq);
        }
        ggml::tensor *codes = cols[0];
        for (int i = 1; i < n_quantizers; ++i)
            codes = ggml::concat(ctx, codes, cols[i], 0);
        return codes;
    }

    ggml::tensor *MOSSResidualLFQ::dequantize(ComputeContext *ctx, ggml::tensor *codes, int k)
    {
        const int64_t T = ggml::get_dim(codes, 1);
        const int n = (k < 0) ? n_quantizers : k;
        ggml::tensor *emb = nullptr;
        for (int i = 0; i < n; ++i)
        {
            ggml::tensor *colv = ggml::view_2d(ctx, codes, 1, T, codes->nb[1], (size_t)i * codes->nb[0]);
            ggml::tensor *col  = ggml::reshape_1d(ctx, ggml::cont(ctx, colv), T);
            ggml::tensor *sel  = ggml::get_rows(ctx, codebooks[i], col);
            ggml::tensor *zq   = out_proj[i]->forward(ctx, sel);
            emb = emb ? ggml::add(ctx, emb, zq) : zq;
        }
        return output_proj.forward(ctx, emb);
    }

    int64_t MOSSResidualLFQ::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += input_proj.get_param_num(effective_only);
        r += output_proj.get_param_num(effective_only);
        for (int i = 0; i < n_quantizers; ++i)
        {
            r += in_proj[i]->get_param_num(effective_only);
            r += out_proj[i]->get_param_num(effective_only);
            r += ggml::nelements(codebooks[i]);
        }
        return r;
    }

    void MOSSResidualLFQ::load(const std::string &path, TensorLoader *loader)
    {
        input_proj.load(path + "input_proj.", loader);
        output_proj.load(path + "output_proj.", loader);
        for (int i = 0; i < n_quantizers; ++i)
        {
            std::string b = path + "quantizers." + std::to_string(i) + ".";
            in_proj[i]->load(b + "in_proj.", loader);
            out_proj[i]->load(b + "out_proj.", loader);
            loader->read_tensor(b + "codebook.weight", codebooks[i]);
        }
    }

    MOSSAudioTokenizer::MOSSAudioTokenizer(InitContext *ctx, const CodecConfig &cfg)
        : cfg(cfg),
          quantizer(ctx, cfg.num_quantizers, cfg.codebook_size, cfg.codebook_dim, cfg.rvq_dim, cfg.code_dim > 0 ? cfg.code_dim : 768)
    {
        const double chan_factor = (cfg.channel_interleave && cfg.channels > 1) ? (double)cfg.channels : 1.0;
        double fr = (double)cfg.sample_rate * chan_factor;

        encoder_stages.resize(cfg.n_enc_stages);
        for (int i = 0; i < cfg.n_enc_stages; ++i)
        {
            TowerStage &s = encoder_stages[i];
            s.cfg.kind       = cfg.enc_kind[i];
            s.cfg.patch_size = cfg.enc_patch[i];
            s.cfg.in_dim     = cfg.enc_in_dim[i];
            s.cfg.d_model    = cfg.enc_d_model[i];
            s.cfg.n_heads    = cfg.enc_n_heads[i];
            s.cfg.n_layers   = cfg.enc_n_layers[i];
            s.cfg.d_ff       = cfg.enc_d_ff[i];
            s.cfg.out_dim    = cfg.enc_out_dim[i];
            s.up             = false;
            if (s.cfg.kind == 0)
            {
                fr /= s.cfg.patch_size;
            }
            else
            {
                s.cfg.context = (int)std::lround(fr * cfg.context_seconds);
                s.transformer.reset(new MOSSTransformerStage(ctx, s.cfg));
            }
        }

        decoder_stages.resize(cfg.n_dec_stages);
        for (int i = 0; i < cfg.n_dec_stages; ++i)
        {
            TowerStage &s = decoder_stages[i];
            s.cfg.kind       = cfg.dec_kind[i];
            s.cfg.patch_size = cfg.dec_patch[i];
            s.cfg.in_dim     = cfg.dec_in_dim[i];
            s.cfg.d_model    = cfg.dec_d_model[i];
            s.cfg.n_heads    = cfg.dec_n_heads[i];
            s.cfg.n_layers   = cfg.dec_n_layers[i];
            s.cfg.d_ff       = cfg.dec_d_ff[i];
            s.cfg.out_dim    = cfg.dec_out_dim[i];
            s.up             = true;
            if (s.cfg.kind == 0)
            {
                fr *= s.cfg.patch_size;
            }
            else
            {
                s.cfg.context = (int)std::lround(fr * cfg.context_seconds);
                s.transformer.reset(new MOSSTransformerStage(ctx, s.cfg));
            }
        }
    }

    int64_t MOSSAudioTokenizer::get_param_num(bool effective_only) const
    {
        int64_t r = quantizer.get_param_num(effective_only);
        for (const auto &s : encoder_stages)
            if (s.transformer) r += s.transformer->get_param_num(effective_only);
        for (const auto &s : decoder_stages)
            if (s.transformer) r += s.transformer->get_param_num(effective_only);
        return r;
    }

    void MOSSAudioTokenizer::load(const std::string &path, TensorLoader *loader)
    {
        for (size_t i = 0; i < encoder_stages.size(); ++i)
        {
            if (encoder_stages[i].transformer)
                encoder_stages[i].transformer->load(path + "encoder." + std::to_string(i) + ".", loader);
        }
        for (size_t i = 0; i < decoder_stages.size(); ++i)
        {
            if (decoder_stages[i].transformer)
                decoder_stages[i].transformer->load(path + "decoder." + std::to_string(i) + ".", loader);
        }
        quantizer.load(path + "quantizer.", loader);
    }

    struct PendingTensorUpload
    {
        ggml::tensor *t;
        std::vector<uint8_t> data;
    };

    static ggml::tensor *make_pos_leaf(ComputeContext *ctx, std::vector<PendingTensorUpload> &pend, int64_t T)
    {
        ggml::tensor *pos = ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_I32, T);
        ggml::set_input(pos);
        PendingTensorUpload p;
        p.t = pos;
        p.data.resize((size_t)T * sizeof(int32_t));
        int32_t *d = (int32_t *)p.data.data();
        for (int64_t i = 0; i < T; ++i) d[i] = (int32_t)i;
        pend.push_back(std::move(p));
        return pos;
    }

    static ggml::tensor *make_mask_leaf(ComputeContext *ctx, std::vector<PendingTensorUpload> &pend, int64_t T, int context)
    {
        ggml::tensor *m = ggml::new_tensor_2d(ctx, ggml::type::GGML_TYPE_F32, T, T);
        ggml::set_input(m);
        PendingTensorUpload p;
        p.t = m;
        p.data.resize((size_t)T * (size_t)T * sizeof(float));
        std::vector<float> vals;
        build_window_mask(vals, (int)T, context);
        std::memcpy(p.data.data(), vals.data(), p.data.size());
        pend.push_back(std::move(p));
        return m;
    }

    bool MOSSAudioTokenizer::encode(TensorGraphEvaluator &eval, const GenerationConfig &gen_config,
                                    const std::vector<float> &pcm_samples, std::vector<int> &codes_out, int &n_frames_out)
    {
        const int64_t factor = (cfg.channel_interleave && cfg.channels > 1) ? cfg.channels : 1;
        const int64_t frame  = (int64_t)cfg.downsample_rate * factor;
        const int64_t n      = (int64_t)pcm_samples.size();
        int64_t padded       = ((n + frame - 1) / frame) * frame;
        if (padded == 0) padded = frame;
        const int64_t T_latent = padded / frame;

        std::vector<PendingTensorUpload> pend;
        const auto make_graph = [this, &pend, padded, &pcm_samples, n](ComputeContext *ctx) -> ggml::tensor * {
            pend.clear();
            ggml::tensor *x = ggml::new_tensor_2d(ctx, ggml::type::GGML_TYPE_F32, padded, 1);
            ggml::set_input(x);
            {
                PendingTensorUpload p;
                p.t = x;
                p.data.assign((size_t)padded * sizeof(float), 0);
                std::memcpy(p.data.data(), pcm_samples.data(), (size_t)n * sizeof(float));
                pend.push_back(std::move(p));
            }

            for (size_t i = 0; i < encoder_stages.size(); ++i)
            {
                const TowerStage &s = encoder_stages[i];
                if (s.cfg.kind == 0)
                {
                    x = patch_down(ctx, x, s.cfg.patch_size);
                }
                else
                {
                    ggml::tensor *h = ggml::cont(ctx, ggml::transpose(ctx, x));
                    const int64_t T = ggml::get_dim(h, 1);
                    ggml::tensor *pos  = make_pos_leaf(ctx, pend, T);
                    ggml::tensor *mask = make_mask_leaf(ctx, pend, T, s.cfg.context);
                    ggml::tensor *y    = s.transformer->forward(ctx, h, pos, mask);
                    x = ggml::cont(ctx, ggml::transpose(ctx, y));
                }
            }
            x = ggml::cont(ctx, ggml::transpose(ctx, x));
            return quantizer.quantize(ctx, x);
        };

        const auto write_input_data = [&pend](ComputeContext *ctx) {
            for (auto &p : pend)
                Backend::write_tensor_data(p.t, p.data.data(), 0, p.data.size());
        };

        std::vector<int64_t> shape;
        std::vector<uint8_t> buf;
        if (!eval.evaluate(gen_config, make_graph, write_input_data, ggml::type::GGML_TYPE_I32, shape, buf))
            return false;

        codes_out.resize(buf.size() / sizeof(int32_t));
        std::memcpy(codes_out.data(), buf.data(), buf.size());
        n_frames_out = (int)T_latent;
        return true;
    }

    bool MOSSAudioTokenizer::decode(TensorGraphEvaluator &eval, const GenerationConfig &gen_config,
                                    const std::vector<int> &codes, int n_frames, int n_vq_used, std::vector<float> &pcm_samples)
    {
        const int64_t NQ = (n_vq_used > 0) ? n_vq_used : cfg.num_quantizers;
        const int64_t T  = n_frames;
        if ((int64_t)codes.size() != NQ * T)
            return false;

        const int k = (n_vq_used > 0 && n_vq_used < cfg.num_quantizers) ? n_vq_used : -1;
        std::vector<PendingTensorUpload> pend;

        const auto make_graph = [this, &pend, &codes, NQ, T, k](ComputeContext *ctx) -> ggml::tensor * {
            pend.clear();
            ggml::tensor *code_t = ggml::new_tensor_2d(ctx, ggml::type::GGML_TYPE_I32, NQ, T);
            ggml::set_input(code_t);
            {
                PendingTensorUpload p;
                p.t = code_t;
                p.data.resize((size_t)(NQ * T) * sizeof(int32_t));
                std::memcpy(p.data.data(), codes.data(), p.data.size());
                pend.push_back(std::move(p));
            }

            ggml::tensor *x = quantizer.dequantize(ctx, code_t, k);
            x = ggml::cont(ctx, ggml::transpose(ctx, x));
            for (size_t i = 0; i < decoder_stages.size(); ++i)
            {
                const TowerStage &s = decoder_stages[i];
                if (s.cfg.kind == 0)
                {
                    x = patch_up(ctx, x, s.cfg.patch_size);
                }
                else
                {
                    ggml::tensor *h = ggml::cont(ctx, ggml::transpose(ctx, x));
                    const int64_t Tt = ggml::get_dim(h, 1);
                    ggml::tensor *pos  = make_pos_leaf(ctx, pend, Tt);
                    ggml::tensor *mask = make_mask_leaf(ctx, pend, Tt, s.cfg.context);
                    ggml::tensor *y    = s.transformer->forward(ctx, h, pos, mask);
                    x = ggml::cont(ctx, ggml::transpose(ctx, y));
                }
            }
            return x;
        };

        const auto write_input_data = [&pend](ComputeContext *ctx) {
            for (auto &p : pend)
                Backend::write_tensor_data(p.t, p.data.data(), 0, p.data.size());
        };

        std::vector<int64_t> shape;
        std::vector<uint8_t> buf;
        if (!eval.evaluate(gen_config, make_graph, write_input_data, ggml::type::GGML_TYPE_F32, shape, buf))
            return false;

        pcm_samples.resize(buf.size() / sizeof(float));
        std::memcpy(pcm_samples.data(), buf.data(), buf.size());
        return true;
    }
}

namespace chatllm::moss::tts
{
    // Local depth transformer layer for v1.5 (GPT-J sequential residual with fused QKV, interleaved RoPE, SiLU MLP)
    class GPTJLocalAttention : public RoPESelfAttention<BaseConsolidatedQKVAttention>
    {
    public:
        GPTJLocalAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int head_dim, int max_length, float rope_theta)
            : RoPESelfAttention<BaseConsolidatedQKVAttention>(ctx, hidden_size, num_attention_heads, num_attention_heads, head_dim, max_length, true, true)
        {
            this->freq_base = rope_theta;
            this->rope_mode = RoPEMode::Interleaved;
        }

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past) override
        {
            const int hidden_size = (int)ggml::get_dim(hidden_states, 0);
            const int qlen        = (int)ggml::get_dim(hidden_states, 1);

            before_forward(ctx, n_past, qlen);

            ggml::tensor *qkv = query_key_value.forward(ctx, hidden_states);

            ggml::tensor *tmpq = ggml::cont(ctx, ggml::view_2d(ctx, qkv, hidden_size, qlen, qkv->nb[1], 0));
            ggml::tensor *tmpk = ggml::cont(ctx, ggml::view_2d(ctx, qkv, hidden_size, qlen, qkv->nb[1], hidden_size * ggml::element_size(qkv)));
            ggml::tensor *tmpv = ggml::cont(ctx, ggml::view_2d(ctx, qkv, hidden_size, qlen, qkv->nb[1], 2 * hidden_size * ggml::element_size(qkv)));

            ggml::tensor *scores = cross_attention(ctx, hidden_size, n_past, qlen, tmpq, tmpk, tmpv);
            return dense.forward(ctx, scores);
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            Block::load(path, loader);
            query_key_value.load(path + "c_attn.", loader);
            dense.load(path + "c_proj.", loader);
        }
    };

    class GPTJLocalMLP : public TheMLP
    {
    public:
        GPTJLocalMLP(InitContext *ctx, int hidden_size, int intermediate_size, ActFunc act = ActFunc::SILU)
            : TheMLP(ctx, hidden_size, intermediate_size, act, true)
        {
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            Block::load(path, loader);
            fc0.load(path + "fc_in.", loader);
            fc1.load(path + "fc_out.", loader);
        }
    };

    class GPTJLocalBlock : public Block
    {
    public:
        GPTJLocalBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                       int head_dim, int max_length, float rope_theta, float ln_eps, ActFunc act = ActFunc::SILU)
            : ln_1(ctx, hidden_size, true, ln_eps),
              attn(ctx, hidden_size, num_attention_heads, head_dim, max_length, rope_theta),
              ln_2(ctx, hidden_size, true, ln_eps),
              mlp(ctx, hidden_size, intermediate_size, act)
        {
        }

        using Block::forward;
        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past) override
        {
            ggml::tensor *residual = hidden_states;
            ggml::tensor *h = ln_1.forward(ctx, hidden_states);
            h = attn.forward(ctx, h, n_past);
            hidden_states = ggml::add(ctx, residual, h);

            residual = hidden_states;
            h = ln_2.forward(ctx, hidden_states);
            h = mlp.forward(ctx, h);
            hidden_states = ggml::add(ctx, residual, h);
            return hidden_states;
        }

        void set_id(int id) override
        {
            Block::set_id(id);
            ln_1.set_id(id);
            attn.set_id(id);
            ln_2.set_id(id);
            mlp.set_id(id);
        }

        size_t get_cache_size(void) const override { return attn.get_cache_size(); }
        void set_cache_buffer(BackendBuffer *buffer) override { attn.set_cache_buffer(buffer); }
        void before_eval(ComputeContext *ctx) override { attn.before_eval(ctx); }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += ln_1.get_param_num(effective_only);
            r += attn.get_param_num(effective_only);
            r += ln_2.get_param_num(effective_only);
            r += mlp.get_param_num(effective_only);
            return r;
        }

        void load(const std::string &path, TensorLoader *loader) override
        {
            Block::load(path, loader);
            ln_1.load(path + "ln_1.", loader);
            attn.load(path + "attn.", loader);
            ln_2.load(path + "ln_2.", loader);
            mlp.load(path + "mlp.", loader);
        }

    public:
        LayerNorm ln_1;
        GPTJLocalAttention attn;
        LayerNorm ln_2;
        GPTJLocalMLP mlp;
    };

    // Local depth transformer layer for v1.0 (Qwen3 with QK RMSNorm and use_rope = false)
    class Qwen3LocalAttention : public QKNormedAttention<RMSNormInplace, RoPESelfAttention<BaseAttention>>
    {
    public:
        typedef QKNormedAttention<RMSNormInplace, RoPESelfAttention<BaseAttention>> Base;
        Qwen3LocalAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int head_dim, int max_length)
            : Base(ctx, hidden_size, num_attention_heads, num_kv_heads, head_dim, max_length, false, false)
        {
            this->use_rope = false;
        }
    };

    class Qwen3LocalBlock : public LMBlock1<RMSNorm, Qwen3LocalAttention, RMSNorm, SiLUMLP>
    {
    public:
        typedef LMBlock1<RMSNorm, Qwen3LocalAttention, RMSNorm, SiLUMLP> Base;
        Qwen3LocalBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                        int num_kv_heads, int head_dim, int max_length, float rms_eps)
            : Base(ctx, hidden_size, num_attention_heads, intermediate_size, num_kv_heads, head_dim, max_length)
        {
            input_layernorm.eps = rms_eps;
            post_attention_layernorm.eps = rms_eps;
            attention.q_layernorm.eps = rms_eps;
            attention.k_layernorm.eps = rms_eps;
        }
    };

    class SpWhitespacePreprocessor : public tokenizer::TextPreprocessor
    {
    public:
        std::string transform(const std::string &s) override
        {
            if (s.empty()) return s;
            std::string out;
            out.reserve(s.size() + 1);
            out.push_back(' ');
            bool in_ws = true;
            for (unsigned char ch : s)
            {
                bool ws = (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '\v' || ch == '\f');
                if (ws)
                {
                    if (!in_ws)
                    {
                        out.push_back(' ');
                        in_ws = true;
                    }
                }
                else
                {
                    out.push_back((char)ch);
                    in_ws = false;
                }
            }
            if (out.size() > 1 && out.back() == ' ')
                out.pop_back();
            return out;
        }
    };

    Tokenizer::Tokenizer(const BaseConfig &cfg)
        : qwen::v3::Tokenizer(cfg),
          is_nano(static_cast<const Config &>(cfg).global_arch == 1)
    {
    }

    size_t Tokenizer::load(tokenizer::DataReader *buffer, int n_vocab)
    {
        if (is_nano)
        {
            tp = new tokenizer::BPEProcessor1();
            size_t r = tp->Load(buffer, n_vocab);
            tp->RegisterPreprocessor(new SpWhitespacePreprocessor());
            tp->AddAddedToken("<user_inst>", 13);
            tp->AddAddedToken("</user_inst>", 14);
            return r;
        }
        size_t r = qwen::v3::Tokenizer::load(buffer, n_vocab);
        return r;
    }

    void Tokenizer::encode(const std::string &text, std::vector<int> &ids) const
    {
        last_text = text;
        tp->Encode(text, &ids);
    }

    static int sample_moss_token(std::vector<float> &logits,
                                 const std::vector<int> &history,
                                 float repetition_penalty,
                                 float top_p,
                                 int top_k,
                                 bool do_sample,
                                 std::mt19937 &rng)
    {
        const int V = (int)logits.size();
        if (V <= 0) return 0;

        if (repetition_penalty > 0.0f && repetition_penalty != 1.0f && !history.empty())
        {
            std::vector<uint8_t> seen(V, 0);
            for (int id : history)
            {
                if (id >= 0 && id < V) seen[id] = 1;
            }
            for (int i = 0; i < V; ++i)
            {
                if (!seen[i]) continue;
                if (logits[i] < 0.0f)
                    logits[i] *= repetition_penalty;
                else
                    logits[i] /= repetition_penalty;
            }
        }

        if (!do_sample)
        {
            int best = 0;
            for (int i = 1; i < V; ++i)
                if (logits[i] > logits[best]) best = i;
            return best;
        }

        struct Cand { int id; float score; };
        std::vector<Cand> cands(V);
        for (int i = 0; i < V; ++i) cands[i] = {i, logits[i]};

        const int K = (top_k > 0 && top_k < V) ? top_k : V;
        if (K < V)
        {
            std::partial_sort(cands.begin(), cands.begin() + K, cands.end(),
                              [](const Cand &a, const Cand &b) {
                                  if (a.score != b.score) return a.score > b.score;
                                  return a.id < b.id;
                              });
            cands.resize(K);
        }
        else
        {
            std::sort(cands.begin(), cands.end(),
                      [](const Cand &a, const Cand &b) {
                          if (a.score != b.score) return a.score > b.score;
                          return a.id < b.id;
                      });
        }

        float m = cands[0].score;
        std::vector<float> probs(cands.size());
        double sum = 0.0;
        for (size_t i = 0; i < cands.size(); ++i)
        {
            probs[i] = std::exp(cands[i].score - m);
            sum += probs[i];
        }
        for (float &p : probs) p = (float)(p / sum);

        if (top_p > 0.0f && top_p < 1.0f)
        {
            double cum = 0.0;
            size_t keep = cands.size();
            for (size_t i = 0; i < cands.size(); ++i)
            {
                cum += probs[i];
                if (cum > (double)top_p && i >= 1)
                {
                    keep = i + 1;
                    break;
                }
            }
            cands.resize(keep);
            probs.resize(keep);
            double rsum = 0.0;
            for (float p : probs) rsum += p;
            for (float &p : probs) p = (float)(p / rsum);
        }

        std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
        return cands[dist(rng)].id;
    }

    LocalGeneration::LocalGeneration(const Config &config, const RuntimeConfig &runtime_config, size_t GRAPH_SIZE)
        : config(config),
          eval(runtime_config, "loc", GRAPH_SIZE),
          _ctx(eval.get_backend_context())
    {
        _ctx.cache_dtype = runtime_config.cache_type;
        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = 2048;
        _ctx.gctx = GGMLContext({.mem_size = num_tensors * tensor_ovhd, .mem_buffer = nullptr, .no_alloc = true});
        _ctx.dtype = config.dtype;

        const int max_local_seq = 64;
        for (int i = 0; i < config.local_num_layers; ++i)
        {
            Block *layer = nullptr;
            if (config.local_arch == 0 || config.local_arch == 2)
            {
                ActFunc act = (config.local_arch == 2) ? ActFunc::GELU : ActFunc::SILU;
                layer = new GPTJLocalBlock(&_ctx,
                                           config.local_hidden_size,
                                           config.local_num_heads,
                                           config.local_ffn_hidden_size,
                                           config.local_head_dim,
                                           max_local_seq,
                                           config.local_rope_theta,
                                           config.local_norm_eps,
                                           act);
            }
            else
            {
                layer = new Qwen3LocalBlock(&_ctx,
                                            config.local_hidden_size,
                                            config.local_num_heads,
                                            config.local_ffn_hidden_size,
                                            config.local_num_kv_heads,
                                            config.local_head_dim,
                                            max_local_seq,
                                            config.local_norm_eps);
            }
            layer->set_id(i);
            auto allocator = _ctx.get_allocator();
            auto buf = allocator->alloc(layer->get_cache_size(), BackendBufAllocator::Usage::Matrix);
            layer->set_cache_buffer(buf);
            layers.add_block(layer);
        }

        if (config.local_arch == 0 || config.local_arch == 2)
        {
            final_norm.reset(new LayerNorm(&_ctx, config.local_hidden_size, true, config.local_norm_eps));
        }
        else
        {
            auto *rn = new RMSNorm(&_ctx, config.local_hidden_size);
            rn->eps = config.local_norm_eps;
            final_norm.reset(rn);
        }

        for (int c = 0; c < config.n_vq; ++c)
        {
            audio_embeddings.emplace_back(new Embedding(&_ctx, config.audio_embed_rows, config.hidden_size));
        }

        const int channels = 1 + config.n_vq;
        if (!config.bare_heads)
        {
            class AdapterSiLUMLP : public Block
            {
            public:
                AdapterSiLUMLP(InitContext *ctx, int in_dim, int mid_dim, int out_dim)
                    : gate_proj(ctx, in_dim, mid_dim, false),
                      down_proj(ctx, mid_dim, out_dim, false),
                      up_proj(ctx, in_dim, mid_dim, false)
                {
                }
                using Block::forward;
                ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states) override
                {
                    ggml::tensor *gate = gate_proj.forward(ctx, hidden_states);
                    gate = ggml::act(ctx, ActFunc::SILU, gate);
                    ggml::tensor *up = up_proj.forward(ctx, hidden_states);
                    ggml::tensor *r  = ggml::mul_inplace(ctx, gate, up);
                    return down_proj.forward(ctx, r);
                }
                int64_t get_param_num(bool effective_only) const override
                {
                    return gate_proj.get_param_num(effective_only) +
                           down_proj.get_param_num(effective_only) +
                           up_proj.get_param_num(effective_only);
                }
                void load(const std::string &path, TensorLoader *loader) override
                {
                    Block::load(path, loader);
                    gate_proj.load(path + "gate_proj.", loader);
                    down_proj.load(path + "down_proj.", loader);
                    up_proj.load(path + "up_proj.", loader);
                }
            public:
                Linear gate_proj;
                Linear down_proj;
                Linear up_proj;
            };

            in_mlp.reset(new AdapterSiLUMLP(&_ctx, config.hidden_size, config.additional_mlp_ffn_size, config.local_hidden_size));
            for (int c = 0; c < channels; ++c)
            {
                out_mlps.emplace_back(new AdapterSiLUMLP(&_ctx, config.local_hidden_size, config.additional_mlp_ffn_size, config.hidden_size));
                auto *hn = new RMSNorm(&_ctx, config.hidden_size);
                hn->eps = config.local_norm_eps;
                head_norms.emplace_back(hn);
            }
        }

        lm_heads.resize(channels);
        if (config.local_text_head_mode == 1)
        {
            local_text_head.reset(new Linear(&_ctx, config.local_hidden_size, 2, false));
        }
        else
        {
            lm_heads[0].reset(new Linear(&_ctx, config.hidden_size, config.vocab_size, false));
        }

        for (int c = 1; c < channels; ++c)
        {
            lm_heads[c].reset(new Linear(&_ctx, config.hidden_size, config.audio_vocab_size, false));
        }
    }

    void LocalGeneration::load(ModelLoader &loader)
    {
        loader.push_allocator_manager(eval.get_layer_allocators());

        for (int i = 0; i < config.local_num_layers; ++i)
        {
            if (config.local_arch == 0 || config.local_arch == 2)
                layers.blocks[i]->load("local_transformer.h." + std::to_string(i) + ".", &loader);
            else
                layers.blocks[i]->load("local_transformer.layers." + std::to_string(i) + ".", &loader);
        }

        if (config.local_arch == 0 || config.local_arch == 2)
            final_norm->load("local_transformer.ln_f.", &loader);
        else
            final_norm->load("local_transformer.norm.", &loader);

        for (int c = 0; c < config.n_vq; ++c)
        {
            audio_embeddings[c]->load("audio_embeddings." + std::to_string(c) + ".", &loader);
        }

        const int channels = 1 + config.n_vq;
        if (!config.bare_heads)
        {
            in_mlp->load("speech_embedding_to_local_mlp.", &loader);
            for (int c = 0; c < channels; ++c)
            {
                out_mlps[c]->load("local_to_speech_embedding_mlps." + std::to_string(c) + ".", &loader);
                head_norms[c]->load("layer_norm_before_lm_heads." + std::to_string(c) + ".", &loader);
            }
        }

        if (config.local_text_head_mode == 1)
        {
            local_text_head->load("local_text_lm_head.", &loader);
        }
        else
        {
            lm_heads[0]->load("lm_heads.0.", &loader);
        }

        for (int c = 1; c < channels; ++c)
        {
            lm_heads[c]->load("audio_lm_heads." + std::to_string(c - 1) + ".", &loader);
        }

        loader.pop_allocator_manager();
        loaded = true;
    }

    bool LocalGeneration::generate_frame(const GenerationConfig &gen_config,
                                         Block *text_embedding,
                                         const float *global_hidden,
                                         float text_temp, float text_top_p, int text_top_k,
                                         float audio_temp, float audio_top_p, int audio_top_k, float audio_rep_penalty,
                                         std::vector<std::vector<int>> &history,
                                         std::vector<int> &next_frame)
    {
        const int channels = 1 + config.n_vq;
        next_frame.assign(channels, 0);

        static thread_local std::mt19937 rng(42);

        if (config.local_text_head_mode == 1)
        {
            // v1.5 binary text head mode:
            // Position 0 takes global_hidden -> h0 -> both local_text_head(h0) [2] and lm_heads[1](h0) [audio_vocab_size].
            ggml::tensor *gh_in = nullptr;
            const auto make_graph_0 = [this, &gh_in](ComputeContext *ctx) -> ggml::tensor * {
                gh_in = ggml::new_tensor_2d(ctx, ggml::type::GGML_TYPE_F32, config.hidden_size, 1);
                ggml::tensor *cur = in_mlp ? in_mlp->forward(ctx, gh_in) : gh_in;
                ggml::tensor *h   = layers.forward(ctx, cur, 0);
                h = final_norm->forward(ctx, h);
                ggml::tensor *dl = local_text_head->forward(ctx, h);
                ggml::tensor *h1 = (out_mlps.size() > 1 && out_mlps[1]) ? out_mlps[1]->forward(ctx, h) : h;
                if (head_norms.size() > 1 && head_norms[1]) h1 = head_norms[1]->forward(ctx, h1);
                ggml::tensor *al = lm_heads[1]->forward(ctx, h1);
                return ggml::concat(ctx, dl, al, 0);
            };
            const auto write_input_0 = [this, &gh_in, global_hidden](ComputeContext *ctx) {
                Backend::write_tensor_data(gh_in, global_hidden, 0, config.hidden_size * sizeof(float));
                layers.before_eval(ctx);
            };

            std::vector<int64_t> shape;
            std::vector<uint8_t> buf;
            if (!eval.evaluate(gen_config, make_graph_0, write_input_0, ggml::type::GGML_TYPE_F32, shape, buf))
                return false;

            const float *out_0 = (const float *)buf.data();
            const int decision_idx = (out_0[1] > out_0[0]) ? 1 : 0;
            const int tok = (decision_idx == 1) ? config.audio_end_token_id : config.audio_assistant_gen_slot_token_id;
            next_frame[0] = tok;
            history[0].push_back(tok);

            std::vector<float> lg0(out_0 + 2, out_0 + 2 + config.audio_vocab_size);
            if (audio_temp > 0.0f)
                for (float &v : lg0) v /= audio_temp;
            int code = sample_moss_token(lg0, history[1], audio_rep_penalty, audio_top_p, audio_top_k, audio_temp > 0.0f, rng);
            next_frame[1] = code;
            history[1].push_back(code);

            for (int ac = 1; ac < config.n_vq; ++ac)
            {
                const int ch = ac + 1;
                const int prev_code = next_frame[ac];
                ggml::tensor *tok_in = nullptr;
                const auto make_graph_ac = [this, &tok_in, ac, ch](ComputeContext *ctx) -> ggml::tensor * {
                    tok_in = ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_I32, 1);
                    ggml::tensor *cur = audio_embeddings[ac - 1]->forward(ctx, tok_in);
                    if (in_mlp) cur = in_mlp->forward(ctx, cur);
                    ggml::tensor *h = layers.forward(ctx, cur, ac);
                    h = final_norm->forward(ctx, h);
                    if (ch < (int)out_mlps.size() && out_mlps[ch]) h = out_mlps[ch]->forward(ctx, h);
                    if (ch < (int)head_norms.size() && head_norms[ch]) h = head_norms[ch]->forward(ctx, h);
                    return lm_heads[ch]->forward(ctx, h);
                };
                const auto write_input_ac = [this, &tok_in, prev_code](ComputeContext *ctx) {
                    Backend::write_tensor_data(tok_in, &prev_code, 0, sizeof(int));
                    layers.before_eval(ctx);
                };

                shape.clear();
                buf.clear();
                if (!eval.evaluate(gen_config, make_graph_ac, write_input_ac, ggml::type::GGML_TYPE_F32, shape, buf))
                    return false;

                const float *ptr = (const float *)buf.data();
                std::vector<float> lg(ptr, ptr + config.audio_vocab_size);
                if (audio_temp > 0.0f)
                    for (float &v : lg) v /= audio_temp;
                int c_code = sample_moss_token(lg, history[ch], audio_rep_penalty, audio_top_p, audio_top_k, audio_temp > 0.0f, rng);
                next_frame[ch] = c_code;
                history[ch].push_back(c_code);
            }
        }
        else if (config.local_text_head_mode == 2)
        {
            // Nano mode: depth 0 samples decision token from tied text_lm_head [vocab_size, hidden_size].
            // If decision != audio_assistant_gen_slot_token_id (9), stop immediately without running depths 1..16.
            ggml::tensor *gh_in = nullptr;
            const auto make_graph_0 = [this, &gh_in](ComputeContext *ctx) -> ggml::tensor * {
                gh_in = ggml::new_tensor_2d(ctx, ggml::type::GGML_TYPE_F32, config.hidden_size, 1);
                ggml::tensor *h = layers.forward(ctx, gh_in, 0);
                h = final_norm->forward(ctx, h);
                return lm_heads[0]->forward(ctx, h);
            };
            const auto write_input_0 = [this, &gh_in, global_hidden](ComputeContext *ctx) {
                Backend::write_tensor_data(gh_in, global_hidden, 0, config.hidden_size * sizeof(float));
                layers.before_eval(ctx);
            };

            std::vector<int64_t> shape;
            std::vector<uint8_t> buf;
            if (!eval.evaluate(gen_config, make_graph_0, write_input_0, ggml::type::GGML_TYPE_F32, shape, buf))
                return false;

            const float *ptr0 = (const float *)buf.data();
            std::vector<float> lg0(ptr0, ptr0 + config.vocab_size);
            if (text_temp > 0.0f)
                for (float &v : lg0) v /= text_temp;
            std::vector<int> empty_hist;
            int decision = sample_moss_token(lg0, empty_hist, 1.0f, text_top_p, text_top_k, text_temp > 0.0f, rng);
            if (decision != config.audio_assistant_gen_slot_token_id)
            {
                next_frame[0] = config.audio_end_token_id;
                return true;
            }
            next_frame[0] = decision;

            for (int c = 0; c < config.n_vq; ++c)
            {
                const int ch = c + 1;
                const int prev_tok = (c == 0) ? decision : next_frame[c];
                ggml::tensor *tok_in = nullptr;
                const auto make_graph_c = [this, &tok_in, c, ch, text_embedding](ComputeContext *ctx) -> ggml::tensor * {
                    tok_in = ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_I32, 1);
                    ggml::tensor *cur = (c == 0) ? text_embedding->forward(ctx, tok_in)
                                                 : audio_embeddings[c - 1]->forward(ctx, tok_in);
                    ggml::tensor *h = layers.forward(ctx, cur, c + 1);
                    h = final_norm->forward(ctx, h);
                    return lm_heads[ch]->forward(ctx, h);
                };
                const auto write_input_c = [this, &tok_in, prev_tok](ComputeContext *ctx) {
                    Backend::write_tensor_data(tok_in, &prev_tok, 0, sizeof(int));
                    layers.before_eval(ctx);
                };

                shape.clear();
                buf.clear();
                if (!eval.evaluate(gen_config, make_graph_c, write_input_c, ggml::type::GGML_TYPE_F32, shape, buf))
                    return false;

                const float *ptr = (const float *)buf.data();
                std::vector<float> lg(ptr, ptr + config.audio_vocab_size);
                if (audio_temp > 0.0f)
                    for (float &v : lg) v /= audio_temp;
                int code = sample_moss_token(lg, history[ch], audio_rep_penalty, audio_top_p, audio_top_k, audio_temp > 0.0f, rng);
                next_frame[ch] = code;
                history[ch].push_back(code);
            }
        }
        else
        {
            // v1.0 full text head + 32 audio heads mode (positions 0..n_vq)
            for (int i = 0; i < channels; ++i)
            {
                ggml::tensor *in_tensor = nullptr;
                const int prev_code = (i > 0) ? next_frame[i - 1] : 0;
                const auto make_graph_i = [this, &in_tensor, i, text_embedding](ComputeContext *ctx) -> ggml::tensor * {
                    ggml::tensor *cur = nullptr;
                    if (i == 0)
                    {
                        in_tensor = ggml::new_tensor_2d(ctx, ggml::type::GGML_TYPE_F32, config.hidden_size, 1);
                        cur = in_tensor;
                    }
                    else
                    {
                        in_tensor = ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_I32, 1);
                        cur = (i == 1) ? text_embedding->forward(ctx, in_tensor)
                                       : audio_embeddings[i - 2]->forward(ctx, in_tensor);
                    }
                    if (in_mlp) cur = in_mlp->forward(ctx, cur);
                    ggml::tensor *h = layers.forward(ctx, cur, i);
                    h = final_norm->forward(ctx, h);
                    if (i < (int)out_mlps.size() && out_mlps[i]) h = out_mlps[i]->forward(ctx, h);
                    if (i < (int)head_norms.size() && head_norms[i]) h = head_norms[i]->forward(ctx, h);
                    return lm_heads[i]->forward(ctx, h);
                };
                const auto write_input_i = [this, &in_tensor, i, global_hidden, prev_code](ComputeContext *ctx) {
                    if (i == 0)
                        Backend::write_tensor_data(in_tensor, global_hidden, 0, config.hidden_size * sizeof(float));
                    else
                        Backend::write_tensor_data(in_tensor, &prev_code, 0, sizeof(int));
                    layers.before_eval(ctx);
                };

                std::vector<int64_t> shape;
                std::vector<uint8_t> buf;
                if (!eval.evaluate(gen_config, make_graph_i, write_input_i, ggml::type::GGML_TYPE_F32, shape, buf))
                    return false;

                const int V = (i == 0) ? config.vocab_size : config.audio_vocab_size;
                const float *ptr = (const float *)buf.data();
                std::vector<float> lg(ptr, ptr + V);
                const float temp = (i == 0) ? text_temp : audio_temp;
                if (temp > 0.0f)
                    for (float &v : lg) v /= temp;
                int code = sample_moss_token(lg, history[i],
                                             (i == 0) ? 1.0f : audio_rep_penalty,
                                             (i == 0) ? text_top_p : audio_top_p,
                                             (i == 0) ? text_top_k : audio_top_k,
                                             temp > 0.0f, rng);
                next_frame[i] = code;
                history[i].push_back(code);
            }
        }
        return true;
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : Base(type, config, runtime_config, 4096 * 4),
          config(config),
          local_gen(config, runtime_config),
          codec_eval(runtime_config, "codec", 65536),
          codec_ctx(codec_eval.get_backend_context())
    {
        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = 32 + config.num_hidden_layers * 20;
        w_ctx_.gctx = GGMLContext({.mem_size = num_tensors * tensor_ovhd, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;

        Block *global_final_norm = nullptr;
        if (config.global_arch == 1)
        {
            global_final_norm = new LayerNorm(LayerMover(&w_ctx_, LayerAllocatorManager::MiscLayer::Epilog),
                                              config.hidden_size, true, 1e-5f);
        }
        else
        {
            global_final_norm = create_final_norm<RMSNorm>(&w_ctx_, config);
        }

        auto create_global_layer = [&](InitContext *ctx, int layer_index) -> Block * {
            if (config.global_arch == 1)
            {
                return new GPTJLocalBlock(ctx,
                                          config.hidden_size,
                                          config.num_attention_heads,
                                          config.intermediate_size,
                                          config.head_dim,
                                          config.max_length,
                                          config.rope_theta,
                                          1e-5f,
                                          ActFunc::GELU);
            }
            else
            {
                auto *layer = new qwen::v3::QWen3Block(ctx,
                                                       config.hidden_size,
                                                       config.num_attention_heads,
                                                       config.intermediate_size,
                                                       config.num_key_value_heads,
                                                       config.head_dim,
                                                       config.max_length);
                layer->attention.freq_base = config.rope_theta;
                return layer;
            }
        };

        transformer = new ModelClass(&w_ctx_,
                                     config.num_hidden_layers,
                                     config.hidden_size,
                                     create_embedding<Embedding>(&w_ctx_, config),
                                     global_final_norm,
                                     nullptr,
                                     create_global_layer);
        transformer->skip_lm_head = true;

        codec_ctx.cache_dtype = runtime_config.cache_type;
        if (config.has_codec)
        {
            const size_t codec_tensors = 4096;
            codec_ctx.gctx = GGMLContext({.mem_size = codec_tensors * tensor_ovhd, .mem_buffer = nullptr, .no_alloc = true});
            codec_ctx.dtype = ggml::type::GGML_TYPE_F32;
            audio_tokenizer.reset(new codec::MOSSAudioTokenizer(&codec_ctx, config.codec_cfg));
        }

        if (config.global_arch == 1)
        {
            text_temperature         = 1.0f;
            text_top_p               = 1.0f;
            text_top_k               = 50;
            audio_temperature        = 0.8f;
            audio_top_p              = 0.95f;
            audio_top_k              = 25;
            audio_repetition_penalty = 1.2f;
        }
        else
        {
            // Default repetition penalty matches v1.0 (1.1) vs v1.5 (1.0)
            audio_repetition_penalty = (config.local_text_head_mode == 1) ? 1.0f : 1.1f;
        }
    }

    void ConditionalGeneration::load(ModelLoader &loader)
    {
        Base::load(loader);
        local_gen.load(loader);
        if (audio_tokenizer && loader.has_tensor("codec.quantizer.input_proj.weight"))
        {
            loader.push_allocator_manager(codec_eval.get_layer_allocators());
            audio_tokenizer->load("codec.", &loader);
            loader.pop_allocator_manager();
            codec_loaded = true;
        }
    }

    void ConditionalGeneration::set_additional_args(const std::map<std::string, std::string> &args)
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        if (tok)
        {
            tok->instruction = utils::get_opt(args, "instruction", tok->instruction);
            tok->language    = utils::get_opt(args, "language", tok->language);
            tok->ref_audio   = utils::get_opt(args, "ref_audio", tok->ref_audio);
        }
        text_temperature         = (float)utils::get_opt(args, "text_temp", text_temperature);
        text_top_p               = (float)utils::get_opt(args, "text_top_p", text_top_p);
        text_top_k               = utils::get_opt(args, "text_top_k", text_top_k);
        audio_temperature        = (float)utils::get_opt(args, "audio_temp", audio_temperature);
        audio_top_p              = (float)utils::get_opt(args, "audio_top_p", audio_top_p);
        audio_top_k              = utils::get_opt(args, "audio_top_k", audio_top_k);
        audio_repetition_penalty = (float)utils::get_opt(args, "audio_rep_penalty", audio_repetition_penalty);
        greedy                   = utils::get_opt(args, "greedy", 0) != 0;
    }

    bool ConditionalGeneration::run_global_step(const GenerationConfig &gen_config,
                                                const std::vector<int> &row_major_ids,
                                                int n_rows,
                                                std::vector<float> &last_hidden_out)
    {
        const int channels = 1 + config.n_vq;
        CHATLLM_CHECK((int)row_major_ids.size() == n_rows * channels);

        if (!initial_run)
        {
            initial_run = true;
            int past = gen_config.max_length / transformer->get_reserved_batch_size() - n_rows;
            if (past < 0) past = 0;
            transformer->custom_embedding = [this, n_rows](ComputeContext *cctx, ggml::tensor *input) -> ggml::tensor * {
                ggml::tensor *sum = transformer->word_embeddings->forward(cctx, input);
                for (int c = 0; c < config.n_vq; ++c)
                {
                    ggml::tensor *idx_t = ggml::new_tensor_1d(cctx, ggml::type::GGML_TYPE_I32, n_rows);
                    ggml::tensor *ae = local_gen.audio_embeddings[c]->forward(cctx, idx_t);
                    sum = ggml::add(cctx, sum, ae);
                }
                return sum;
            };
            bool ok = before_initial_run(n_rows, gen_config, past);
            transformer->custom_embedding = nullptr;
            if (!ok)
                return false;
        }

        // Transpose row-major [n_rows, channels] into per-channel contiguous vectors [channels][n_rows]
        std::vector<std::vector<int>> ch_data(channels, std::vector<int>(n_rows));
        for (int s = 0; s < n_rows; ++s)
        {
            for (int c = 0; c < channels; ++c)
                ch_data[c][s] = row_major_ids[(size_t)s * channels + c];
        }

        ForwardContext ctx(&backend_context);
        ctx.user_options = w_ctx_.user_options;
        ctx.gctx = GGMLContext({.mem_size = backend_context.buf_compute_meta.size(), .mem_buffer = backend_context.buf_compute_meta.data(), .no_alloc = true});
        ctx.gf = ggml::new_graph_custom(&ctx, GRAPH_SIZE, false);

        set_dbg_ctx(&ctx);
        ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Prolog);

        std::vector<ggml::tensor *> ch_tensors(channels, nullptr);
        for (int c = 0; c < channels; ++c)
            ch_tensors[c] = ggml::new_tensor_1d(&ctx, ggml::type::GGML_TYPE_I32, n_rows);

        const auto custom_emb = [this, &ch_tensors](ComputeContext *cctx, ggml::tensor *input) -> ggml::tensor * {
            ggml::tensor *sum = transformer->word_embeddings->forward(cctx, ch_tensors[0]);
            for (int c = 0; c < config.n_vq; ++c)
            {
                ggml::tensor *ae = local_gen.audio_embeddings[c]->forward(cctx, ch_tensors[c + 1]);
                sum = ggml::add(cctx, sum, ae);
            }
            return sum;
        };

        transformer->custom_embedding = custom_emb;
        ggml::tensor *r = transformer->forward(&ctx, ch_tensors[0], n_past);
        transformer->custom_embedding = nullptr;

        ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Epilog);
        ggml::set_output(r);
        ggml::build_forward_expand(&ctx, r);

        if (!ctx.allocate()) return false;

        for (int c = 0; c < channels; ++c)
            Backend::write_tensor_data(ch_tensors[c], ch_data[c].data(), 0, n_rows * sizeof(int));

        transformer->before_eval(&ctx);
        ctx.compute();

        // With skip_lm_head = true, r is [hidden_size, 1] (the final normed hidden state of the last token)
        last_hidden_out.resize(ggml::nbytes(r) / sizeof(float));
        Backend::read_tensor_data(r, last_hidden_out.data());

        ctx.reset();
        n_past += n_rows;
        return true;
    }

    static bool load_wav_file(const char *fn, std::vector<float> &samples, int target_sr)
    {
        std::ifstream file(fn, std::ios::binary);
        if (!file.is_open()) return false;
        char riff[4];
        file.read(riff, 4);
        if (std::memcmp(riff, "RIFF", 4) != 0) return false;
        uint32_t file_size = 0; file.read((char *)&file_size, 4);
        char wave[4]; file.read(wave, 4);
        if (std::memcmp(wave, "WAVE", 4) != 0) return false;

        uint16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
        uint32_t sample_rate = 0;
        std::vector<char> raw_data;

        while (file)
        {
            char chunk_id[4];
            uint32_t chunk_size = 0;
            if (!file.read(chunk_id, 4)) break;
            if (!file.read((char *)&chunk_size, 4)) break;
            if (std::memcmp(chunk_id, "fmt ", 4) == 0)
            {
                file.read((char *)&audio_format, 2);
                file.read((char *)&num_channels, 2);
                file.read((char *)&sample_rate, 4);
                uint32_t byte_rate = 0; file.read((char *)&byte_rate, 4);
                uint16_t block_align = 0; file.read((char *)&block_align, 2);
                file.read((char *)&bits_per_sample, 2);
                if (chunk_size > 16) file.seekg(chunk_size - 16, std::ios::cur);
            }
            else if (std::memcmp(chunk_id, "data", 4) == 0)
            {
                raw_data.resize(chunk_size);
                file.read(raw_data.data(), chunk_size);
                break;
            }
            else
            {
                file.seekg(chunk_size, std::ios::cur);
            }
        }
        if (raw_data.empty() || num_channels == 0 || bits_per_sample == 0) return false;

        size_t bytes_per_sample = bits_per_sample / 8;
        size_t n_samples = raw_data.size() / (num_channels * bytes_per_sample);
        std::vector<float> mono(n_samples);
        if (audio_format == 1 && bits_per_sample == 16)
        {
            const int16_t *p = (const int16_t *)raw_data.data();
            for (size_t i = 0; i < n_samples; ++i)
            {
                float sum = 0.0f;
                for (int c = 0; c < num_channels; ++c)
                    sum += p[i * num_channels + c] / 32768.0f;
                mono[i] = sum / num_channels;
            }
        }
        else if (audio_format == 3 && bits_per_sample == 32)
        {
            const float *p = (const float *)raw_data.data();
            for (size_t i = 0; i < n_samples; ++i)
            {
                float sum = 0.0f;
                for (int c = 0; c < num_channels; ++c)
                    sum += p[i * num_channels + c];
                mono[i] = sum / num_channels;
            }
        }
        else return false;

        if ((int)sample_rate == target_sr)
        {
            samples = std::move(mono);
        }
        else
        {
            size_t out_len = (size_t)((double)mono.size() * target_sr / sample_rate);
            samples.resize(out_len);
            for (size_t i = 0; i < out_len; ++i)
            {
                double src_idx = (double)i * sample_rate / target_sr;
                size_t idx0 = (size_t)src_idx;
                size_t idx1 = std::min(idx0 + 1, mono.size() - 1);
                float frac = (float)(src_idx - idx0);
                samples[i] = mono[idx0] * (1.0f - frac) + mono[idx1] * frac;
            }
        }
        return true;
    }

    void ConditionalGeneration::speech_synthesis(const GenerationConfig &gen_config, const std::vector<int> &input_ids,
                                                 std::vector<int16_t> &audio, int &sample_rate, int &channels)
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
        CHATLLM_CHECK(tok != nullptr);

        const int n_vq   = config.n_vq;
        const int n_ch   = 1 + n_vq;
        const int pad_id = config.audio_pad_code;

        sample_rate = config.codec_cfg.sample_rate > 0 ? config.codec_cfg.sample_rate : config.sample_rate;
        channels    = (config.codec_cfg.channel_interleave && config.codec_cfg.channels > 1) ? config.codec_cfg.channels : 1;

        auto t_synth_start = std::chrono::high_resolution_clock::now();

        // 1. Optional reference audio encoding (voice cloning)
        std::vector<int> ref_codes;
        int T_ref = 0;
        double ref_duration_sec = 0.0;
        if (!tok->ref_audio.empty() && audio_tokenizer && codec_loaded)
        {
            std::vector<float> ref_mono;
            bool ok = audio::load(tok->ref_audio.c_str(), ref_mono, sample_rate);
            if (!ok)
                ok = load_wav_file(tok->ref_audio.c_str(), ref_mono, sample_rate);
            if (ok)
            {
                ref_duration_sec = (double)ref_mono.size() / sample_rate;
                std::vector<float> ref_pcm;
                if (channels == 2)
                {
                    ref_pcm.resize(ref_mono.size() * 2);
                    for (size_t i = 0; i < ref_mono.size(); ++i)
                    {
                        ref_pcm[2 * i + 0] = ref_mono[i];
                        ref_pcm[2 * i + 1] = ref_mono[i];
                    }
                }
                else
                {
                    ref_pcm = std::move(ref_mono);
                }
                audio_tokenizer->encode(codec_eval, gen_config, ref_pcm, ref_codes, T_ref);
            }
        }

        auto t_ref_done = std::chrono::high_resolution_clock::now();

        // 2. Build (S, 1 + n_vq) row-major prompt IDs
        std::vector<int> prompt_ids;
        auto push_row = [&](int text_id, const int *audio_row) {
            prompt_ids.push_back(text_id);
            if (audio_row)
                prompt_ids.insert(prompt_ids.end(), audio_row, audio_row + n_vq);
            else
                for (int c = 0; c < n_vq; ++c) prompt_ids.push_back(pad_id);
        };
        auto push_text = [&](const std::string &s) {
            std::vector<int> e;
            tok->tp->Encode(s, &e);
            for (int t : e) push_row(t, nullptr);
        };

        const std::string after_ref =
            "\n- Instruction:\n" + tok->instruction +
            "\n- Tokens:\nNone"
            "\n- Quality:\nNone"
            "\n- Sound Event:\nNone"
            "\n- Ambient Sound:\nNone"
            "\n- Language:\n" + tok->language +
            "\n- Text:\n";

        push_row(config.im_start_token_id, nullptr);
        push_text("user\n");
        push_text("<user_inst>\n- Reference(s):\n");
        if (T_ref <= 0)
        {
            push_text("None");
        }
        else
        {
            push_row(config.audio_start_token_id, nullptr);
            for (int r = 0; r < T_ref; ++r)
                push_row(config.audio_user_slot_token_id, ref_codes.data() + (size_t)r * n_vq);
            push_row(config.audio_end_token_id, nullptr);
        }
        push_text(after_ref);
        for (int t : input_ids) push_row(t, nullptr);
        push_text("\n</user_inst>");
        push_row(config.im_end_token_id, nullptr);
        push_text("\n");
        push_row(config.im_start_token_id, nullptr);
        push_text("assistant\n");
        push_row(config.audio_start_token_id, nullptr);

        const int S = (int)(prompt_ids.size() / n_ch);
        if (S <= 0) return;

        auto t_prefill_start = std::chrono::high_resolution_clock::now();

        // 3. Global backbone prefill
        n_past = 0;
        n_past_offset = 0;
        std::vector<float> gh;
        if (!run_global_step(gen_config, prompt_ids, S, gh))
            return;

        auto t_prefill = std::chrono::high_resolution_clock::now();

        // 4. Seed per-channel repetition penalty history (v1.5 / v1.0 seed with prompt tokens; Nano starts empty)
        const int max_steps = gen_config.max_new_tokens > 0 ? gen_config.max_new_tokens : 512;
        std::vector<std::vector<int>> history(n_ch);
        for (int c = 0; c < n_ch; ++c)
        {
            history[c].reserve((size_t)S + max_steps);
            if (config.global_arch != 1)
            {
                for (int s = 0; s < S; ++s)
                    history[c].push_back(prompt_ids[(size_t)s * n_ch + c]);
            }
        }

        const float t_temp = greedy ? 0.0f : text_temperature;
        const float a_temp = greedy ? 0.0f : audio_temperature;

        // 5. Time x Depth autoregressive generation loop
        std::vector<int> gen_audio;
        int n_steps = 0;
        bool stopped = false;

        for (int step = 0; step < max_steps; ++step)
        {
            std::vector<int> next_frame;
            if (!local_gen.generate_frame(gen_config,
                                          transformer->word_embeddings,
                                          gh.data(),
                                          t_temp, text_top_p, text_top_k,
                                          a_temp, audio_top_p, audio_top_k, audio_repetition_penalty,
                                          history,
                                          next_frame))
            {
                break;
            }

            for (int c = 1; c < n_ch; ++c)
                gen_audio.push_back(next_frame[c]);
            ++n_steps;

            if (next_frame[0] == config.audio_end_token_id)
            {
                stopped = true;
                break;
            }

            if (!run_global_step(gen_config, next_frame, 1, gh))
                break;
        }

        auto t_gen = std::chrono::high_resolution_clock::now();

        int T = n_steps;
        if (stopped && T > 0) --T;
        if (T <= 0 || !audio_tokenizer || !codec_loaded)
            return;

        // 6. Decode audio codes via MOSS-Audio-Tokenizer
        std::vector<int> codes_to_decode(gen_audio.begin(), gen_audio.begin() + (size_t)T * n_vq);
        std::vector<float> wav;
        if (!audio_tokenizer->decode(codec_eval, gen_config, codes_to_decode, T, n_vq, wav) || wav.empty())
            return;

        auto t_decode = std::chrono::high_resolution_clock::now();

        // 7. Loudness normalization to -20.0 dBFS (gain clamped to [-3.0, +3.0] dB)
        {
            const float target_dbfs = -20.0f;
            double sumsq = 0.0;
            for (float v : wav) sumsq += (double)v * v;
            const float rms = (float)std::sqrt(sumsq / wav.size() + 1e-9);
            const float current_dbfs = 20.0f * std::log10(rms);
            float gain = std::max(-3.0f, std::min(3.0f, target_dbfs - current_dbfs));
            const float factor = std::pow(10.0f, gain / 20.0f);
            for (float &v : wav) v *= factor;
        }

        // Convert float [-1, 1] PCM to int16_t
        audio.resize(wav.size());
        for (size_t i = 0; i < wav.size(); ++i)
        {
            float clamped = std::max(-1.0f, std::min(1.0f, wav[i]));
            audio[i] = (int16_t)std::lround(clamped * 32767.0f);
        }

        auto t_synth_end = std::chrono::high_resolution_clock::now();
        double d_total = std::chrono::duration<double>(t_synth_end - t_synth_start).count();
        double d_ref = std::chrono::duration<double>(t_ref_done - t_synth_start).count();
        double d_prefill = std::chrono::duration<double>(t_prefill - t_prefill_start).count();
        double d_loop = std::chrono::duration<double>(t_gen - t_prefill).count();
        double d_codec = std::chrono::duration<double>(t_decode - t_gen).count();
        double audio_sec = (channels > 0 && sample_rate > 0) ? (double)audio.size() / (channels * sample_rate) : 0.0;
        double rtf = audio_sec > 0.0 ? (d_total / audio_sec) : 0.0;
        double ms_per_frame = T > 0 ? (d_loop * 1000.0 / T) : 0.0;

        std::cout << "\n================ [MOSS-TTS Benchmark Summary] ================\n";
        if (T_ref > 0)
        {
            std::cout << "  [Voice Clone Mode]   : Enabled\n";
            std::cout << "  Reference Audio      : " << std::fixed << std::setprecision(2) << ref_duration_sec << " s (" << T_ref << " latent frames)\n";
            std::cout << "  Ref Audio Encode time: " << std::fixed << std::setprecision(3) << d_ref * 1000.0 << " ms\n";
        }
        else
        {
            std::cout << "  [Voice Clone Mode]   : Disabled (Standard TTS)\n";
        }
        std::cout << "  Prompt length        : " << S << " tokens / frames (including reference audio if any)\n";
        std::cout << "  Generated audio      : " << T << " latent frames (" << std::fixed << std::setprecision(2) << audio_sec << " s, " << sample_rate << " Hz)\n";
        std::cout << "  Prefill time         : " << std::fixed << std::setprecision(3) << d_prefill * 1000.0 << " ms\n";
        std::cout << "  Generation loop time : " << std::fixed << std::setprecision(3) << d_loop * 1000.0 << " ms ("
                  << std::setprecision(2) << ms_per_frame << " ms/frame, " << (d_loop > 0.0 ? (T / d_loop) : 0.0) << " fps)\n";
        std::cout << "  Codec decode time    : " << std::fixed << std::setprecision(3) << d_codec * 1000.0 << " ms\n";
        std::cout << "  ------------------------------------------------------------\n";
        std::cout << "  Pure Model Execution : " << std::fixed << std::setprecision(3) << d_total << " s (excluding model loading)\n";
        std::cout << "  Real-Time Factor(RTF): " << std::fixed << std::setprecision(3) << rtf << " ("
                  << (rtf < 1.0 ? "FASTER than Real-Time!" : "Slower than Real-Time") << ")\n";
        std::cout << "==============================================================\n" << std::endl;
    }

    REGISTER_MODEL_LOADER(MOSS_TTS, chatllm::moss::tts, 1);
}
