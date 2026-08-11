"""Forward hook registration for extracting hidden representations from encoder/decoder layers."""

from collections import defaultdict


class HookManager:
    """Registers forward hooks on encoder and decoder layers and collects their outputs."""

    def __init__(self):
        self._handles = []
        self._representations = defaultdict(list)

    def register_hooks(self, model):
        """Hook every encoder/decoder layer (all five architectures expose .layers)."""
        for i, layer in enumerate(model.encoder.layers):
            handle = layer.register_forward_hook(self._make_hook("encoder", i))
            self._handles.append(handle)

        for i, layer in enumerate(model.decoder.layers):
            handle = layer.register_forward_hook(self._make_hook("decoder", i))
            self._handles.append(handle)

    def _make_hook(self, layer_type, layer_index):
        key = (layer_type, layer_index)

        def hook_fn(module, input, output):
            self._representations[key].append(output.detach().cpu())

        return hook_fn

    def get_representations(self):
        """Dict of (layer_type, layer_index) to list of [seq_len, batch, embed_dim] tensors."""
        return dict(self._representations)

    def clear(self):
        self._representations.clear()

    def remove_hooks(self):
        """Remove all hooks and clear collected representations."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._representations.clear()
