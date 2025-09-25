/**
 * GPU-Accelerated HFT Engine
 * Uses WebGL and WebGPU for parallel computation of arbitrage opportunities
 * Optimized for maximum throughput in high-frequency trading scenarios
 */

class GPUAcceleratedHFTEngine {
    constructor() {
        this.gl = null;
        this.gpu = null;
        this.device = null;
        
        // Computation capabilities
        this.webglSupported = false;
        this.webgpuSupported = false;
        
        // Performance tracking
        this.performanceMetrics = {
            computeShaderTime: [],
            dataTransferTime: [],
            totalPipelineTime: []
        };
        
        this.initializeGPUAcceleration();
    }
    
    async initializeGPUAcceleration() {
        console.log('🚀 Initializing GPU acceleration for HFT...');
        
        // Initialize WebGPU (preferred for compute)
        await this.initializeWebGPU();
        
        // Initialize WebGL (fallback for older browsers)
        this.initializeWebGL();
        
        console.log(`✅ GPU acceleration initialized: WebGPU=${this.webgpuSupported}, WebGL=${this.webglSupported}`);
    }
    
    async initializeWebGPU() {
        if (!navigator.gpu) {
            console.warn('⚠️ WebGPU not supported in this browser');
            return;
        }
        
        try {
            const adapter = await navigator.gpu.requestAdapter({
                powerPreference: 'high-performance'
            });
            
            if (!adapter) {
                throw new Error('No WebGPU adapter found');
            }
            
            this.device = await adapter.requestDevice();
            this.webgpuSupported = true;
            
            // Create compute pipeline for arbitrage detection
            await this.createArbitrageComputePipeline();
            
            console.log('✅ WebGPU initialized for high-performance computing');
            
        } catch (error) {
            console.warn('⚠️ WebGPU initialization failed:', error);
        }
    }
    
    async createArbitrageComputePipeline() {
        // WebGPU compute shader for parallel arbitrage detection
        const shaderCode = `
            struct PriceData {
                exchange1: f32,
                exchange2: f32,
                exchange3: f32,
                timestamp: f32,
            }
            
            struct ArbitrageOpportunity {
                profit: f32,
                confidence: f32,
                latency: f32,
                volume: f32,
            }
            
            @group(0) @binding(0) var<storage, read> priceData: array<PriceData>;
            @group(0) @binding(1) var<storage, read_write> opportunities: array<ArbitrageOpportunity>;
            @group(0) @binding(2) var<uniform> params: struct {
                minProfit: f32,
                maxLatency: f32,
                dataCount: u32,
                timestamp: f32,
            };
            
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let index = global_id.x;
                if (index >= params.dataCount) { return; }
                
                let data = priceData[index];
                
                // Calculate triangular arbitrage profit
                // Path: Exchange1 -> Exchange2 -> Exchange3 -> Exchange1
                let rate1 = 1.0 / data.exchange1;
                let rate2 = data.exchange2;
                let rate3 = 1.0 / data.exchange3;
                
                let finalAmount = rate1 * rate2 * rate3;
                let profit = (finalAmount - 1.0) * 100.0; // Profit percentage
                
                // Calculate confidence based on price stability
                let priceVariance = abs(data.exchange1 - data.exchange2) + 
                                  abs(data.exchange2 - data.exchange3) + 
                                  abs(data.exchange3 - data.exchange1);
                let confidence = 1.0 / (1.0 + priceVariance * 0.01);
                
                // Calculate estimated latency based on timestamp
                let latency = params.timestamp - data.timestamp;
                
                // Estimate tradeable volume (simplified model)
                let avgPrice = (data.exchange1 + data.exchange2 + data.exchange3) / 3.0;
                let volume = 1000000.0 / avgPrice; // $1M equivalent volume
                
                // Only store profitable opportunities within latency constraints
                if (profit > params.minProfit && latency < params.maxLatency) {
                    opportunities[index] = ArbitrageOpportunity(
                        profit,
                        confidence,
                        latency,
                        volume
                    );
                } else {
                    opportunities[index] = ArbitrageOpportunity(0.0, 0.0, 0.0, 0.0);
                }
            }
        `;
        
        this.arbitrageShaderModule = this.device.createShaderModule({
            code: shaderCode
        });
        
        this.arbitrageComputePipeline = this.device.createComputePipeline({
            compute: {
                module: this.arbitrageShaderModule,
                entryPoint: 'main'
            }
        });
        
        console.log('✅ WebGPU arbitrage compute pipeline created');
    }
    
    initializeWebGL() {
        const canvas = document.createElement('canvas');
        canvas.width = 1024;
        canvas.height = 1024;
        
        this.gl = canvas.getContext('webgl2');
        
        if (!this.gl) {
            console.warn('⚠️ WebGL2 not supported');
            return;
        }
        
        this.webglSupported = true;
        
        // Create shaders for parallel computation
        this.createWebGLShaders();
        
        console.log('✅ WebGL2 initialized for parallel computation');
    }
    
    createWebGLShaders() {
        // Vertex shader (pass-through)
        const vertexShaderSource = `#version 300 es
            precision highp float;
            
            in vec2 a_position;
            out vec2 v_texCoord;
            
            void main() {
                gl_Position = vec4(a_position, 0.0, 1.0);
                v_texCoord = (a_position + 1.0) / 2.0;
            }
        `;
        
        // Fragment shader for arbitrage computation
        const fragmentShaderSource = `#version 300 es
            precision highp float;
            
            uniform sampler2D u_priceTexture;
            uniform vec2 u_resolution;
            uniform float u_minProfit;
            uniform float u_currentTime;
            
            in vec2 v_texCoord;
            out vec4 fragColor;
            
            void main() {
                vec2 coord = v_texCoord;
                
                // Sample price data (R=exchange1, G=exchange2, B=exchange3, A=timestamp)
                vec4 priceData = texture(u_priceTexture, coord);
                
                float price1 = priceData.r;
                float price2 = priceData.g;
                float price3 = priceData.b;
                float timestamp = priceData.a;
                
                // Skip if invalid data
                if (price1 <= 0.0 || price2 <= 0.0 || price3 <= 0.0) {
                    fragColor = vec4(0.0);
                    return;
                }
                
                // Calculate triangular arbitrage
                float rate1 = 1.0 / price1;
                float rate2 = price2;
                float rate3 = 1.0 / price3;
                
                float finalAmount = rate1 * rate2 * rate3;
                float profit = (finalAmount - 1.0) * 100.0;
                
                // Calculate latency
                float latency = u_currentTime - timestamp;
                
                // Calculate confidence
                float priceSpread = abs(price1 - price2) + abs(price2 - price3) + abs(price3 - price1);
                float confidence = 1.0 / (1.0 + priceSpread * 0.01);
                
                // Output: R=profit, G=confidence, B=latency, A=volume
                if (profit > u_minProfit) {
                    float volume = 1000000.0 / ((price1 + price2 + price3) / 3.0);
                    fragColor = vec4(profit, confidence, latency, volume);
                } else {
                    fragColor = vec4(0.0);
                }
            }
        `;
        
        this.webglProgram = this.createShaderProgram(
            this.gl, 
            vertexShaderSource, 
            fragmentShaderSource
        );
        
        // Create buffers for computation
        this.setupWebGLBuffers();
    }
    
    createShaderProgram(gl, vertexSource, fragmentSource) {
        const vertexShader = this.loadShader(gl, gl.VERTEX_SHADER, vertexSource);
        const fragmentShader = this.loadShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
        
        const program = gl.createProgram();
        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);
        gl.linkProgram(program);
        
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            console.error('Failed to link shader program:', gl.getProgramInfoLog(program));
            gl.deleteProgram(program);
            return null;
        }
        
        return program;
    }
    
    loadShader(gl, type, source) {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.error('Shader compilation error:', gl.getShaderInfoLog(shader));
            gl.deleteShader(shader);
            return null;
        }
        
        return shader;
    }
    
    setupWebGLBuffers() {
        const gl = this.gl;
        
        // Create quad vertices for full-screen computation
        const vertices = new Float32Array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0
        ]);
        
        this.vertexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
        
        // Create framebuffer for render-to-texture
        this.framebuffer = gl.createFramebuffer();
        
        console.log('✅ WebGL buffers and framebuffer created');
    }
    
    // Main GPU-accelerated arbitrage detection
    async detectArbitrageOpportunitiesGPU(priceData, options = {}) {
        const startTime = performance.now();
        
        const config = {
            minProfit: 0.01, // 0.01% minimum profit
            maxLatency: 100, // 100ms maximum latency
            ...options
        };
        
        let opportunities;
        
        if (this.webgpuSupported) {
            opportunities = await this.computeArbitrageWebGPU(priceData, config);
        } else if (this.webglSupported) {
            opportunities = this.computeArbitrageWebGL(priceData, config);
        } else {
            // Fallback to CPU computation
            opportunities = this.computeArbitrageCPU(priceData, config);
        }
        
        const endTime = performance.now();
        const computeTime = endTime - startTime;
        
        this.performanceMetrics.totalPipelineTime.push(computeTime);
        
        return {
            opportunities: opportunities.filter(opp => opp.profit > 0),
            computeTime,
            processedCount: priceData.length,
            throughput: priceData.length / (computeTime / 1000), // ops per second
            accelerationMethod: this.webgpuSupported ? 'WebGPU' : 
                               this.webglSupported ? 'WebGL' : 'CPU'
        };
    }
    
    async computeArbitrageWebGPU(priceData, config) {
        const device = this.device;
        const dataCount = priceData.length;
        
        // Create input buffer
        const inputData = new Float32Array(dataCount * 4); // 4 floats per entry
        for (let i = 0; i < dataCount; i++) {
            inputData[i * 4] = priceData[i].exchange1 || 0;
            inputData[i * 4 + 1] = priceData[i].exchange2 || 0;
            inputData[i * 4 + 2] = priceData[i].exchange3 || 0;
            inputData[i * 4 + 3] = priceData[i].timestamp || Date.now();
        }
        
        const inputBuffer = device.createBuffer({
            size: inputData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        device.queue.writeBuffer(inputBuffer, 0, inputData);
        
        // Create output buffer
        const outputBuffer = device.createBuffer({
            size: dataCount * 4 * 4, // 4 floats per result
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        
        // Create uniform buffer for parameters
        const uniformData = new Float32Array([
            config.minProfit,
            config.maxLatency,
            dataCount,
            Date.now()
        ]);
        
        const uniformBuffer = device.createBuffer({
            size: uniformData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        device.queue.writeBuffer(uniformBuffer, 0, uniformData);
        
        // Create bind group
        const bindGroup = device.createBindGroup({
            layout: this.arbitrageComputePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: inputBuffer } },
                { binding: 1, resource: { buffer: outputBuffer } },
                { binding: 2, resource: { buffer: uniformBuffer } }
            ]
        });
        
        // Execute compute shader
        const commandEncoder = device.createCommandEncoder();
        const computePass = commandEncoder.beginComputePass();
        
        computePass.setPipeline(this.arbitrageComputePipeline);
        computePass.setBindGroup(0, bindGroup);
        computePass.dispatchWorkgroups(Math.ceil(dataCount / 256));
        computePass.end();
        
        // Read back results
        const readBuffer = device.createBuffer({
            size: outputBuffer.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        
        commandEncoder.copyBufferToBuffer(
            outputBuffer, 0,
            readBuffer, 0,
            outputBuffer.size
        );
        
        device.queue.submit([commandEncoder.finish()]);
        
        await readBuffer.mapAsync(GPUMapMode.READ);
        const resultData = new Float32Array(readBuffer.getMappedRange());
        
        // Convert results back to objects
        const opportunities = [];
        for (let i = 0; i < dataCount; i++) {
            const offset = i * 4;
            const profit = resultData[offset];
            
            if (profit > 0) {
                opportunities.push({
                    profit: resultData[offset],
                    confidence: resultData[offset + 1],
                    latency: resultData[offset + 2],
                    volume: resultData[offset + 3],
                    index: i
                });
            }
        }
        
        readBuffer.unmap();
        
        return opportunities;
    }
    
    computeArbitrageWebGL(priceData, config) {
        const gl = this.gl;
        const dataCount = priceData.length;
        
        // Create texture from price data
        const textureSize = Math.ceil(Math.sqrt(dataCount));
        const textureData = new Float32Array(textureSize * textureSize * 4);
        
        for (let i = 0; i < dataCount; i++) {
            textureData[i * 4] = priceData[i].exchange1 || 0;
            textureData[i * 4 + 1] = priceData[i].exchange2 || 0;
            textureData[i * 4 + 2] = priceData[i].exchange3 || 0;
            textureData[i * 4 + 3] = priceData[i].timestamp || Date.now();
        }
        
        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texImage2D(
            gl.TEXTURE_2D, 0, gl.RGBA32F, 
            textureSize, textureSize, 0,
            gl.RGBA, gl.FLOAT, textureData
        );
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        
        // Setup render target
        const resultTexture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, resultTexture);
        gl.texImage2D(
            gl.TEXTURE_2D, 0, gl.RGBA32F,
            textureSize, textureSize, 0,
            gl.RGBA, gl.FLOAT, null
        );
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
        gl.framebufferTexture2D(
            gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
            gl.TEXTURE_2D, resultTexture, 0
        );
        
        // Render computation
        gl.viewport(0, 0, textureSize, textureSize);
        gl.useProgram(this.webglProgram);
        
        // Set uniforms
        gl.uniform1i(gl.getUniformLocation(this.webglProgram, 'u_priceTexture'), 0);
        gl.uniform2f(gl.getUniformLocation(this.webglProgram, 'u_resolution'), textureSize, textureSize);
        gl.uniform1f(gl.getUniformLocation(this.webglProgram, 'u_minProfit'), config.minProfit);
        gl.uniform1f(gl.getUniformLocation(this.webglProgram, 'u_currentTime'), Date.now());
        
        // Bind vertex data
        const positionLocation = gl.getAttribLocation(this.webglProgram, 'a_position');
        gl.enableVertexAttribArray(positionLocation);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
        gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);
        
        // Execute computation
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        
        // Read back results
        const resultData = new Float32Array(textureSize * textureSize * 4);
        gl.readPixels(
            0, 0, textureSize, textureSize,
            gl.RGBA, gl.FLOAT, resultData
        );
        
        // Convert to opportunities array
        const opportunities = [];
        for (let i = 0; i < dataCount; i++) {
            const offset = i * 4;
            const profit = resultData[offset];
            
            if (profit > 0) {
                opportunities.push({
                    profit: resultData[offset],
                    confidence: resultData[offset + 1],
                    latency: resultData[offset + 2],
                    volume: resultData[offset + 3],
                    index: i
                });
            }
        }
        
        return opportunities;
    }
    
    // CPU fallback computation
    computeArbitrageCPU(priceData, config) {
        const opportunities = [];
        const currentTime = Date.now();
        
        for (let i = 0; i < priceData.length; i++) {
            const data = priceData[i];
            
            if (!data.exchange1 || !data.exchange2 || !data.exchange3) continue;
            
            const rate1 = 1.0 / data.exchange1;
            const rate2 = data.exchange2;
            const rate3 = 1.0 / data.exchange3;
            
            const finalAmount = rate1 * rate2 * rate3;
            const profit = (finalAmount - 1.0) * 100;
            
            if (profit > config.minProfit) {
                const latency = currentTime - (data.timestamp || currentTime);
                const avgPrice = (data.exchange1 + data.exchange2 + data.exchange3) / 3;
                const confidence = 1.0 / (1.0 + Math.abs(data.exchange1 - data.exchange2) * 0.01);
                
                opportunities.push({
                    profit,
                    confidence,
                    latency,
                    volume: 1000000 / avgPrice,
                    index: i
                });
            }
        }
        
        return opportunities;
    }
    
    // Performance monitoring and optimization
    getPerformanceMetrics() {
        return {
            computeShader: this.calculateStats(this.performanceMetrics.computeShaderTime),
            dataTransfer: this.calculateStats(this.performanceMetrics.dataTransferTime),
            totalPipeline: this.calculateStats(this.performanceMetrics.totalPipelineTime),
            accelerationCapabilities: {
                webgpu: this.webgpuSupported,
                webgl: this.webglSupported
            }
        };
    }
    
    calculateStats(array) {
        if (array.length === 0) return { avg: 0, min: 0, max: 0 };
        
        const sorted = [...array].sort((a, b) => a - b);
        const sum = sorted.reduce((acc, val) => acc + val, 0);
        
        return {
            avg: sum / array.length,
            min: sorted[0],
            max: sorted[sorted.length - 1],
            p95: sorted[Math.floor(sorted.length * 0.95)],
            count: array.length
        };
    }
    
    // Benchmark different acceleration methods
    async benchmark(sampleSize = 10000) {
        console.log('🏁 Running GPU acceleration benchmarks...');
        
        // Generate test data
        const testData = Array.from({length: sampleSize}, (_, i) => ({
            exchange1: Math.random() * 100 + 50,
            exchange2: Math.random() * 100 + 50,
            exchange3: Math.random() * 100 + 50,
            timestamp: Date.now() - Math.random() * 1000
        }));
        
        const results = {};
        
        // Benchmark WebGPU
        if (this.webgpuSupported) {
            const startTime = performance.now();
            await this.computeArbitrageWebGPU(testData, {});
            results.webgpu = performance.now() - startTime;
        }
        
        // Benchmark WebGL
        if (this.webglSupported) {
            const startTime = performance.now();
            this.computeArbitrageWebGL(testData, {});
            results.webgl = performance.now() - startTime;
        }
        
        // Benchmark CPU
        const startTime = performance.now();
        this.computeArbitrageCPU(testData, {});
        results.cpu = performance.now() - startTime;
        
        console.log('📊 GPU Benchmark Results:', results);
        return results;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = GPUAcceleratedHFTEngine;
}

console.log('📦 GPU-Accelerated HFT Engine module loaded');