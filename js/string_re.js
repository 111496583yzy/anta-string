import { app } from "../../scripts/app.js";

app.registerExtension({
	name: "StringRe.DynamicPorts",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "ImageDominantColor") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                const node = this;
                
                // Find the max_colors widget
                const widget = this.widgets.find(w => w.name === "max_colors");
                if (widget) {
                    // Save original callback
                    const originalCallback = widget.callback;
                    widget.callback = function(v) {
                        if (originalCallback) {
                            originalCallback.apply(this, arguments);
                        }
                        
                        // 2 fixed outputs (palette, list) + N dynamic outputs
                        const fixedCount = 2;
                        const targetCount = fixedCount + v;
                        
                        // Current outputs
                        const currentCount = node.outputs ? node.outputs.length : 0;
                        
                        if (currentCount < targetCount) {
                            // Add missing outputs
                            for (let i = currentCount; i < targetCount; i++) {
                                const hexIndex = i - fixedCount + 1;
                                node.addOutput(`hex_${hexIndex}`, "STRING");
                            }
                        } else if (currentCount > targetCount) {
                            // Remove extra outputs
                            // We need to remove from the end
                            // removeOutput expects index
                            // We loop backwards
                            for (let i = currentCount - 1; i >= targetCount; i--) {
                                node.removeOutput(i);
                            }
                        }
                        
                        // Resize node to fit new outputs if needed (optional, Comfy usually handles it but sometimes needs a nudge)
                        if (node.onResize) {
                            node.onResize(node.size);
                        }
                    };
                    
                    // Trigger once to set initial state
                    widget.callback(widget.value);
                }

				return r;
			};
		}
	},
});
