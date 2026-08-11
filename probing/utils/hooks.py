"""Forward hook registration for extracting hidden representations from encoder/decoder layers."""

import logging
from collections import defaultdict

logger = logging.getLogger("probing.extract")


class HookManager:
    """Registers forward hooks on encoder and decoder layers and collects their outputs."""

    def __init__(self):
        self._handles = []
        self._representations = defaultdict(list)

    def register_hooks(self, model):
        """Register forward hooks on all encoder and decoder layers.

        Works identically for all five architectures (Vanilla,
        Character-separated, Feature-invariant, Independent-feature,
        Feature-geometric): every model subclasses transformer.Transformer and
        exposes encoder/decoder modules with a ``.layers[i]`` list.

        Args:
            model: A loaded Transformer, TagTransformer, IndependentFeatureTransformer,
                or BinaryFeatureTransformer model.
        """
        for i, layer in enumerate(model.encoder.layers):
            handle = layer.register_forward_hook(self._make_hook("encoder", i))
            self._handles.append(handle)
        logger.info("Registered hooks on %d encoder layers", len(model.encoder.layers))

        for i, layer in enumerate(model.decoder.layers):
            handle = layer.register_forward_hook(self._make_hook("decoder", i))
            self._handles.append(handle)
        logger.info("Registered hooks on %d decoder layers", len(model.decoder.layers))

    def _make_hook(self, layer_type, layer_index):
        """Create a hook callback that stores detached CPU tensors.

        Args:
            layer_type: "encoder" or "decoder".
            layer_index: Zero-based layer index.
        """
        key = (layer_type, layer_index)

        def hook_fn(module, input, output):
            self._representations[key].append(output.detach().cpu())

        return hook_fn

    def get_representations(self):
        """Return collected representations as a dict mapping (layer_type, layer_index) to list of tensors.

        Each tensor has shape [seq_len, batch_size, embed_dim].
        """
        return dict(self._representations)

    def clear(self):
        """Clear collected representations for the next forward pass or batch."""
        self._representations.clear()

    def remove_hooks(self):
        """Remove all registered forward hooks and clear collected data."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._representations.clear()
        logger.info("Removed all forward hooks")
