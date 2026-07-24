# Surface formation literature, 2020–2026

## Reading rule and verdict frame

This is an annotated bibliography for a deterministic, symbolic, law-governed implicit-surface creature kernel: authored specification in, compiled body out, with typed obstruction verdicts at every boundary. The authors are frontier LLMs operating through a transactional session surface. That changes the selection rule. A paper is useful here when it supplies a compact vocabulary, a deterministic evaluator, a certifiable bound, a typed failure signal, a compositional interface, or a feedback protocol that an LLM can learn and repair. A visually convincing neural output without a stable semantic program is a derived view, not an authority.

The date window is inclusive: 2020 through 2026. “Venue” distinguishes peer-reviewed publication from an arXiv or vendor release. Web links are primary paper, project, or vendor sources; access checked 2026-07-22.

## 1. Procedural creatures, animals, and part-based parameterizations

### 1.1 Infinigen: procedural natural-world generation

**Citation.** Alexander Raistrick, Lahav Lipson, Zeyu Ma, Lingjie Mei, Mingzhe Wang, Yiming Zuo, Karhan Kayan, Hongyu Wen, Beining Han, Yihan Wang, Alejandro Newell, Hei Law, Ankit Goyal, Kaiyu Yang, and Jia Deng. “Infinite Photorealistic Worlds Using Procedural Generation.” *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2023, pp. 12630–12641. [Project and paper](https://infinigen.org/).

**What it does.** Infinigen generates natural-world scenes entirely from randomized mathematical rules rather than external asset scans. Its coverage includes plants, animals, terrain, and natural phenomena, and its outputs include real geometric detail and automatic ground-truth annotations. The creature generators are assembled from structured parts and parameters: the released system exposes class-specific generators, configurable randomness, and a compact genome-like control surface from which many bodies can be sampled. Its purpose is broad photorealistic synthetic data, not a theorem prover for anatomy.

**Relevance under the kernel constraints.** Adopt the separation between a finite part grammar and a derived render, not the uncontrolled randomness. A creature genome should be a typed, canonical value whose fields select body plan, part counts, attachment sites, profiles, material layers, and seed; compilation should be pure once the seed is fixed. Infinigen is teachable to LLM authors because parts and parameters are inspectable, but its randomized Blender/Python execution is not by itself a deterministic acceptance contract. Require canonical seed derivation, stable traversal, and typed rejection for an impossible body plan, unresolved attachment, or out-of-range profile. The “genome” is an excellent authored input; the mesh is only its view.

### 1.2 Infinigen Indoors: the same procedural architecture beyond nature

**Citation.** Alexander Raistrick, Lingjie Mei, Karhan Kayan, David Yan, Yiming Zuo, Beining Han, Hongyu Wen, Meenal Parakh, Stamatis Alexandropoulos, Lahav Lipson, Zeyu Ma, and Jia Deng. “Infinigen Indoors: Photorealistic Indoor Scenes Using Procedural Generation.” *CVPR*, 2024, pp. 21783–21794. [Project bibliography](https://infinigen.org/).

**What it does.** This successor broadens Infinigen’s procedural system to indoor scenes, with generators for rooms, furniture, materials, lighting, and object arrangements. Its contribution is less about animal morphology than about reusable function-oriented procedural generation: assets are generated from rules, composed into scenes, and rendered with consistent annotations. It demonstrates that the core method is a generator ecosystem rather than a single animal model.

**Relevance under the kernel constraints.** The useful lesson is a typed generator registry whose entries own their parameter vocabulary and output contract. Do not copy a Blender node graph as a second semantic surface. Instead, expose a closed `BodyPlan` family and derive geometry, material, and annotation views from it. For LLM authors, this is preferable to asking for raw mesh code: a model can select a named generator and fill validated parameters, while compiler feedback names the exact failed field or interface. The danger is global scene orchestration and implicit defaults; every derived asset must retain its source constructor and deterministic seed in the receipt.

### 1.3 Infinigen-Articulated / Infinigen-Sim

**Citation.** Abhishek Joshi, Beining Han, Jack Nugent, Max Gonzalez Saez-Diez, Yiming Zuo, Jonathan Liu, Hongyu Wen, Stamatis Alexandropoulos, Karhan Kayan, Anna Calveri, Tao Sun, Gaowen Liu, Yi Shao, Alexander Raistrick, and Jia Deng. “Procedural Generation of Articulated Simulation-Ready Assets.” *arXiv preprint arXiv:2505.10755*, 2025. [Paper and code](https://arxiv.org/abs/2505.10755).

**What it does.** Infinigen-Articulated extends the procedural approach from static assets to articulated, simulation-ready objects. Users define individual parts, joints, axes, limits, and continuous and discrete variation parameters; the toolkit exports assets to common robotics simulators. The paper reports generators with millions to (10^{20}) possible instances per category and evaluates segmentation, policy generalization, and sim-to-real transfer. It is not a creature-specific musculature system, but it makes articulation metadata part of procedural generation rather than a post-hoc rigging pass.

**Relevance under the kernel constraints.** This is directly relevant to a body compiler: joints, attachments, degrees of freedom, and collision roles should be generated from the same authoritative part graph as the surface. Adopt the explicit relation between part identity and joint identity, but replace Blender node state with immutable typed records. LLM authors benefit from a small articulation DSL whose errors can say `JointAxisObstruction`, `MissingAttachment`, or `NonManifoldPart`, then request a local repair. The reported asset diversity is useful evidence for parameterized genomes; it is not evidence that arbitrary combinations are lawful. Compile only after graph compatibility and kinematic bounds descend successfully.

### 1.4 ProcFunc: function-oriented procedural 3D generation

**Citation.** Alexander Raistrick, Karhan Kayan, Jack Nugent, David Yan, Lingjie Mei, Meenal Parakh, Hongyu Wen, Dylan Li, Yiming Zuo, Erich Liang, and Jia Deng. “ProcFunc: Function-Oriented Abstractions for Procedural 3D Generation in Python.” *arXiv preprint arXiv:2604.26943*, 2026. [Paper and project entry](https://infinigen.org/).

**What it does.** ProcFunc presents function-oriented abstractions for procedural 3D generation, continuing the Infinigen line toward reusable, composable generator functions rather than ad hoc scene scripts. The 2026 release is an arXiv primary source rather than a settled conference result, so its claims should be treated as current research direction. Its architectural emphasis is valuable: procedural assets are programs with parameters, reusable functions, and composition boundaries.

**Relevance under the kernel constraints.** This is the closest match to an LLM-authored symbolic body surface, provided the Python surface is narrowed into a typed kernel DSL. Function-oriented composition gives authors named constructors, local sections, and an obvious place to attach proof obligations. The kernel should borrow the algebraic shape, not Python’s ambient dynamism: closed parameter types, canonical ordering, no hidden global state, and an interpreter that emits either a field or an obstruction. This is high-value adoption for teachability and low-value adoption as a runtime dependency until its determinism and failure semantics are independently sealed.

### 1.5 AWOL: language control of parametric animal models

**Citation.** Silvia Zuffi and Michael J. Black. “AWOL: Analysis WithOut synthesis using Language.” *European Conference on Computer Vision (ECCV)*, 2024. [Project and paper](https://awol.is.tue.mpg.de/).

**What it does.** AWOL maps language or image embeddings into parameters of existing parametric models. For animals it uses a statistical quadruped model; for trees it uses a procedural generator. A small set of shape-text pairs trains the mapping, and the authors show generation or estimation of animals not explicitly seen during training. The output remains an explicit mesh with a known parametric source, unlike unconstrained text-to-3D synthesis.

**Relevance under the kernel constraints.** AWOL confirms the correct LLM role: infer a candidate parameter section, not author arbitrary geometry. The learned mapping itself is not deterministic or guaranteed, but its target is a teachable parameter space. In the kernel, language can propose `GenomePatch` values; the compiler then validates body-plan legality, profile bounds, and interface compatibility. Rendered feedback can help an LLM repair a proposal, but only typed compiler verdicts can authorize it. A learned latent-to-parameter map belongs at the session boundary and must never replace the symbolic genome or fabricate a successful body.

### 1.6 Reconstructing Animals and the Wild

**Citation.** Peter Kulits, Michael J. Black, and Silvia Zuffi. “Reconstructing Animals and the Wild.” *CVPR*, 2025, pp. 16565–16577. [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/html/Kulits_Reconstructing_Animals_and_the_Wild_CVPR_2025_paper.html).

**What it does.** RAW reconstructs a natural scene from one image as a structured composition of animals and environmental objects. It uses an autoregressive model to decode image features into a compositional scene representation and builds a million-image synthetic dataset using tools introduced by Infinigen. The representation is deliberately object-based and low-dimensional: the method recovers approximate animal and environment structure rather than a photogrammetric truth surface.

**Relevance under the kernel constraints.** RAW is evidence that structured asset selection and composition scale better than asking a model to hallucinate one monolithic mesh. Its learned decoder is not suitable as the authoritative compiler, but it can propose a scene or body-plan section. Adopt the representation principle: ordered named parts, explicit scene relations, and derived geometry. For LLM authors, a rendered multi-view diagnostic can expose silhouette or placement errors; a typed obstruction can expose missing part, invalid relation, or unsupported class. Keep the learned prior outside the deterministic descent and require a canonical symbolic program before formation.

## 2. LLM/VLM authoring of geometry through programs

### 2.1 3D-GPT and the BlenderGPT family

**Citation.** Chunyi Sun, Junlin Han, Weijian Deng, Xinlong Wang, Zishan Qin, and Stephen Gould. “3D-GPT: Procedural 3D Modeling with Large Language Models.” *arXiv preprint arXiv:2310.12945*, 2023. [Paper](https://arxiv.org/abs/2310.12945). BlenderGPT is a public Blender assistant/project family, not a single stable peer-reviewed paper; the name is used for multiple implementations.

**What it does.** 3D-GPT divides instruction-driven modeling into task dispatch, conceptualization, and modeling agents. The system expands a terse request into a richer procedural description, extracts parameters, and invokes Blender to create the result. The BlenderGPT family demonstrates the simpler version of this idea: an LLM emits Blender Python or invokes Blender operations. Both approaches expose the central empirical problem: code can compile while the spatial result is wrong.

**Relevance under the kernel constraints.** The agent decomposition is useful only if the modeling agent targets a closed symbolic vocabulary. A GOLEM authoring session should accept a typed patch or constructor program, not arbitrary Blender Python. The compiler should return syntax/type errors, geometric witnesses, and rendered views as separate feedback channels. LLMs can reliably fill parameter slots and repair local failures when the error names a constructor and field; they are far less reliable at maintaining hidden global scene state. Use 3D-GPT as evidence for a transactional authoring surface, not as permission to make Blender the semantic owner.

### 2.2 SceneCraft: planning, execution, rendering, review

**Citation.** Ziniu Hu, Ahmet Iscen, Aashi Jain, Thomas Kipf, Yisong Yue, David A. Ross, Cordelia Schmid, and Alireza Fathi. “SceneCraft: An LLM Agent for Synthesizing 3D Scene as Blender Code.” *ICLR 2024 Workshop on AGI*, 2024. [OpenReview paper](https://openreview.net/forum?id=JgWm2Xu6UT).

**What it does.** SceneCraft builds an asset list and scene graph, writes Blender code for numerical layout constraints, renders the scene, and uses a vision-language reviewer to critique and revise it. It also learns reusable script functions rather than repeatedly asking the model to invent the same operations. The reported gains come from decomposition, explicit relational planning, executable code, and an image-based refinement loop, not from a single perfect first generation.

**Relevance under the kernel constraints.** This is the clearest evidence for a session protocol with local descent and visual/compiler feedback. Adopt the blueprint–program–render–review shape, but make the blueprint a typed section and make review outputs typed observations or obstructions. A VLM can say that a shoulder silhouette is too broad; the kernel must translate that into a bounded parameter delta or reject the request as under-specified. Library learning should produce canonical constructors owned by the kernel, never an untracked second library. The render is a derived diagnostic; the accepted body remains the compiled symbolic result.

### 2.3 DeepCAD: CAD as a construction sequence

**Citation.** Rundi Wu, Chang Xiao, and Changxi Zheng. “DeepCAD: A Deep Generative Network for Computer-Aided Design Models.” *ICCV*, 2021, pp. 6772–6782. [Project and paper](https://www.cs.columbia.edu/cg/deepcad/).

**What it does.** DeepCAD models a CAD object as a sequence of operations with parameters, rather than as a voxel grid, point cloud, or final mesh. A Transformer learns to autoencode and generate these construction sequences, using a dataset of 178,238 CAD models. The output is editable because it preserves a construction history, but the learned sequence can still be invalid, geometrically unsuitable, or semantically unlike the prompt.

**Relevance under the kernel constraints.** DeepCAD supplies the right representational instinct: author a program whose execution derives geometry. The kernel should be stricter than token-level sequence generation by giving each constructor a typed input and output, tracking references through stable IDs, and validating every intermediate interface. An LLM can learn a finite grammar much more readily than a numerical mesh encoding, especially when the compiler returns the first failed operation and its expected witness. Do not import DeepCAD’s neural decoder into the core; import the construction-history idea and retain deterministic interpretation.

### 2.4 SkexGen: disentangled construction codebooks

**Citation.** Xiang Xu, Karl D. D. Willis, Joseph G. Lambourne, Chin-Yi Cheng, Pradeep Kumar Jayaraman, and Yasutaka Furukawa. “SkexGen: Autoregressive Generation of CAD Construction Sequences with Disentangled Codebooks.” *Proceedings of the 39th International Conference on Machine Learning (ICML)*, PMLR 162, 2022, pp. 24698–24724. [Project](https://samxuxiang.github.io/skexgen/).

**What it does.** SkexGen generates sketch-and-extrude CAD sequences with separate codebooks for topology, geometry, and extrusion variation. The disentanglement improves diversity and user control in the learned design space. It remains a neural generator over a constrained construction representation; the codebooks are statistical controls, not formal types, and the paper’s output is still subject to execution and validity limits.

**Relevance under the kernel constraints.** The conceptual transfer is to separate semantic axes in the authored genome: topology, dimensions, profile shape, and tissue policy should not be one undifferentiated token stream. This gives LLMs localized repair targets and gives the compiler independent obstruction classes. A deterministic kernel can use a closed product of typed axes with lawful recombination rules, while an optional learned model proposes values. Do not interpret disentangled latent coordinates as guarantees. The authoritative object is the validated construction program; codebook proposals are merely sections awaiting descent.

### 2.5 Text2CAD: natural language to parametric construction

**Citation.** Mohammad Sadil Khan, Sankalp Sinha, Talha Uddin Sheikh, Didier Stricker, Sk Aziz Ali, and Muhammad Zeshan Afzal. “Text2CAD: Generating Sequential CAD Designs from Beginner-to-Expert Level Text Prompts.” *NeurIPS 37*, 2024, pp. 7552–7579. [Proceedings paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0e5b96f97c1813bb75f6c28532c2ecc7-Abstract-Conference.html).

**What it does.** Text2CAD creates roughly 170,000 models and 660,000 text annotations from the DeepCAD corpus, ranging from abstract instructions to detailed dimensions. Its autoregressive Transformer generates sequential parametric CAD models and evaluates visual quality, parametric precision, and geometric accuracy. The system demonstrates that natural-language difficulty can be reduced by translating language into a construction sequence with explicit numeric and topological parameters.

**Relevance under the kernel constraints.** This is directly teachable: accept beginner-level natural language at the session edge, but lower it into typed constructors before any surface is formed. The annotations suggest a useful authoring ladder from intent to exact profile, attachment, and dimension constraints. The missing piece for GOLEM is typed obstruction handling: invalid references, impossible thickness, and incompatible tissue interfaces must stop compilation. The LLM should be allowed to revise a local constructor in response to that evidence; it should not be allowed to “repair” by changing unrelated parts or by emitting a plausible mesh.

### 2.6 CADCodeVerify: visual/compiler feedback refinement

**Citation.** Kamel Alrashedy, Pradyumna Tambwekar, Zulfiqar Haider Zaidi, Megan Langwasser, Wei Xu, and Matthew Gombolay. “Generating CAD Code with Vision-Language Models for 3D Designs.” *ICLR*, 2025. [Proceedings paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/81a934cd364e18ea6fdeaf57a93c17d4-Abstract-Conference.html).

**What it does.** CADCodeVerify generates validation questions from the user’s requirements, renders or inspects the CAD result with a VLM, answers those questions, and feeds corrective feedback back into code generation. CADPrompt contains 200 prompts paired with expert scripting code. On its reported GPT-4 evaluation, the loop reduced point-cloud distance by 7.30% and improved compile/success rate by roughly 5% relative to prior prompting. The important result is not that a VLM becomes a verifier; it is that execution and visual evidence improve the next program.

**Relevance under the kernel constraints.** This is the strongest direct precedent for the transactional author surface. Split feedback into compiler facts, geometric measurements, and rendered perceptual observations; give the LLM a repair target with an address and an allowed edit scope. VLM judgements remain non-authoritative because they can be wrong, but they are useful abductive hints. The kernel should turn accepted hints into typed constraints and reject unsupported claims. Adopt the loop, not the illusion of visual verification: a successful render is not proof of topology, thickness, or law satisfaction.

### 2.7 CAD-Coder and code-trained VLMs

**Citation.** Anna C. Doris, Md Ferdous Alam, Amin Heyrani Nobari, and Faez Ahmed. “CAD-Coder: An Open-Source Vision-Language Model for Computer-Aided Design Code Generation.” *arXiv preprint arXiv:2505.14646*, 2025. [Paper and code](https://arxiv.org/abs/2505.14646). Related 2025 work: “CAD-Coder: Text-to-CAD Generation with Chain-of-Thought and Geometric Reward,” *arXiv preprint arXiv:2505.19713*, 2025.

**What it does.** CAD-Coder fine-tunes a VLM to emit editable CadQuery Python from images, using the 163,000-pair GenCAD-Code dataset. The paper reports a 100% valid-syntax rate on its benchmark and strong solid-similarity results, with some generalization to real images and unseen operations. The related text-to-CAD line adds chain-of-thought and geometric reward signals. These are code-generation systems with geometric evaluation, not formal CAD proof systems.

**Relevance under the kernel constraints.** The 100% syntax result is precisely why syntax is the wrong stopping point. Code-trained models are promising authors for a typed DSL because they learn operation order and parameter patterns, but the kernel must validate semantics: watertightness, part identity, field bounds, and layer compatibility. A CadQuery-like surface could be useful only as a derived adapter, not a parallel semantic owner. Prefer direct generation of the existing body language, with geometric rewards projected into named typed verdicts. The useful frontier is code plus evidence; code alone is still decorative syntax.

## 3. Neural implicit and hybrid representations with deterministic residue

### 3.1 LipMLP: Lipschitz-regularized smooth fields

**Citation.** Hsueh-Ti Derek Liu, Francis Williams, Alec Jacobson, Sanja Fidler, and Or Litany. “Learning Smooth Neural Functions via Lipschitz Regularization.” *SIGGRAPH ’22 Conference Proceedings*, 2022, 13 pages. [Paper and project](https://research.nvidia.com/labs/toronto-ai/lip-mlp/).

**What it does.** LipMLP penalizes an upper bound on a neural field’s Lipschitz constant to improve smooth interpolation, extrapolation, and partial reconstruction from point clouds. The method addresses smoothness in a latent descriptor and coordinate-conditioned neural function; it does not promise exact signed distance, deterministic training, or an interpretable construction history.

**Relevance under the kernel constraints.** The portable idea is a certified or conservatively bounded field slope. A symbolic primitive can carry a Lipschitz bound and use it for safe sphere tracing, local blend support, interval pruning, and obstruction witnesses. The trained MLP itself should remain outside the core because initialization, optimization, hardware, and model weights compromise reproducibility and semantic auditability. For LLM authors, `FieldBound` is teachable and actionable; “the network looks smooth” is not. Adopt the bound as metadata and validation law, not the neural field as body authority.

### 3.2 MeshSDF and DeepMesh: differentiable extraction as a fitting view

**Citation.** Benoit Guillard, Edoardo Remelli, Artem Lukoianov, Stephan R. Richter, Timur Bagautdinov, and Pascal Fua. “DeepMesh: Differentiable Iso-Surface Extraction.” *arXiv preprint arXiv:2106.11795*, 2021; precursor “MeshSDF: Differentiable Iso-Surface Extraction,” *arXiv preprint arXiv:2006.03997*, 2020. [DeepMesh paper](https://arxiv.org/abs/2106.11795).

**What it does.** These methods differentiate the location of surface samples extracted from an implicit field, avoiding a non-differentiable Marching Cubes boundary in optimization pipelines. They support inverse reconstruction and physically driven shape optimization, and allow topology-changing mesh outputs from learned implicit fields. Their target is gradient-based fitting and learning, not exact constructive semantics.

**Relevance under the kernel constraints.** Differentiable extraction is useful as an offline proposal mechanism: optimize profile or blend parameters against a target render, then emit the resulting parameter patch into the symbolic compiler. It must not become an acceptance path because gradients do not establish that a field is a valid SDF, watertight, or law-governed. The correct boundary is `FitProposal -> typed validation -> deterministic compile`. If optimization produces a non-regular level set, disconnected component, or out-of-vocabulary primitive, return an obstruction and retain the authored program unchanged.

### 3.3 SNARF: neural implicit skinning

**Citation.** Xu Chen, Yufeng Zheng, Michael J. Black, Otmar Hilliges, and Andreas Geiger. “SNARF: Differentiable Forward Skinning for Animating Non-Rigid Neural Implicit Shapes.” *ICCV*, 2021. [Project and paper](https://xuchen-ethz.github.io/snarf/).

**What it does.** SNARF combines linear-blend-skinning intuition with an implicit neural shape. Given a deformed point, it finds canonical correspondences through iterative root finding and learns a forward deformation field without direct correspondence supervision. It provides continuous, resolution-independent neural surfaces and supports novel poses, but the representation and learned correspondences are model-dependent and numerically solved.

**Relevance under the kernel constraints.** SNARF is evidence that implicit skinning remains technically productive, not evidence that it should own a deterministic body. The transferable boundary is a correspondence problem with explicit roots, branch counts, and residuals. A symbolic kernel could use root-finding only for a bounded derived skin view, with a fixed iteration budget and `AmbiguousCorrespondence` or `RootResidual` verdicts. The LLM should author bone and layer relations, never neural weights. If correspondence cannot be proved unique enough for the requested guarantee, compilation stops rather than selecting a convenient branch.

### 3.4 DiffCSG: differentiable CSG without mesh booleans

**Citation.** Haocheng Yuan, Adrien Bousseau, Hao Pan, Chengquan Zhang, Niloy J. Mitra, and Changjian Li. “DiffCSG: Differentiable CSG via Rasterization.” *SIGGRAPH Asia Conference Papers*, 2024, pp. 1–10. DOI [10.1145/3680528.3687608](https://doi.org/10.1145/3680528.3687608).

**What it does.** DiffCSG renders CSG models through rasterization rather than black-box mesh processing and adds antialiasing at primitive intersections so image gradients can optimize CSG parameters. It supports direct and image-based editing of CSG primitives. The representation remains explicit and interpretable at the CSG level, but the differentiable rasterizer is an optimization instrument, not a proof that the optimized tree satisfies all geometric contracts.

**Relevance under the kernel constraints.** This is a strong offline fitting tool for symbolic vocabularies. A target image can propose primitive dimensions, placement, or operator choices; the authoritative compiler can then re-evaluate the resulting CSG or implicit tree exactly and issue typed verdicts. The boundary is especially clear for LLM authors: image feedback can suggest “move the eye socket,” while the kernel decides whether the patch preserves containment and topology. Adopt differentiable CSG as a proposal interpreter or test oracle, never as a silent runtime optimizer in the transactional commit path.

### 3.5 Lipschitz Pruning: exact and conservative SDF acceleration

**Citation.** Adrien Bousseau and Alexandre Dai. “Lipschitz Pruning: Hierarchical Simplification of Primitive-Based SDFs.” *Computer Graphics Forum*, 44(2), 2025. [Paper](https://wbrbr.org/publications/LipschitzPruning/documents/LipschitzPruning_submitted_to_EG25_lowres.pdf).

**What it does.** Lipschitz Pruning simplifies primitive-based SDF trees locally by proving that a subtree cannot affect a region, replacing operators with operands or constants. It handles exact and lower-bound/conservative SDFs, including hard and smooth CSG operators, and reports real-time sphere-tracing speedups for large animated trees. The method’s key invariant is not visual similarity but regional equivalence under a Lipschitz bound.

**Relevance under the kernel constraints.** This is one of the best 2020s adoptions for a deterministic field kernel. Store a lower-bound field and a Lipschitz witness on every primitive and composition node; use the witness both to prune evaluation and to explain why an operand was irrelevant in an obstruction receipt. It also gives LLM authors compact feedback: “this field cannot influence the requested junction” is a semantic fact, not a vague render complaint. Preserve the distinction between exact and conservative fields; a conservative bound can accelerate and validate, but must not be advertised as exact geometry.

## 4. Muscle, fascia, tissue, and production deformation

### 4.1 EMU: efficient heterogeneous muscle simulation

**Citation.** Vismay Modi, Lawson Fulton, Alec Jacobson, Shinjiro Sueda, and David I. W. Levin. “EMU: Efficient Muscle Simulation in Deformation Space.” *Computer Graphics Forum*, 40(1), 234–248, 2021. DOI [10.1111/cgf.14185](https://doi.org/10.1111/cgf.14185).

**What it does.** EMU simulates heterogeneous musculoskeletal motion without geometric coarsening, combining soft muscle, stiff tendon, bone, and joints in one deformation-space model. It is designed to scale better than conventional FEM with element count and to parallelize more easily while remaining visually close to FEM results. It is a forward simulation method, not a surface formation grammar.

**Relevance under the kernel constraints.** EMU supplies a valuable optional interpreter for an already formed volumetric layer. Its heterogeneous material boundary maps naturally to typed muscle, tendon, and bone sections, while its deformation result remains a derived view. The pure core should construct the problem and validate material and attachment laws; an effect boundary may run the solver under sealed settings and return residuals. For LLM authors, use named material policies and attachment constraints, not raw solver matrices. Non-convergence, inverted elements, and incompatible material joins are typed obstructions, never cached geometry with a red warning.

### 4.2 Volume Preserving Simulation of Soft Tissue with Skin

**Citation.** Seung Heon Sheen, Egor Larionov, and Dinesh K. Pai. “Volume Preserving Simulation of Soft Tissue with Skin.” *Proceedings of the ACM on Computer Graphics and Interactive Techniques*, 4(3), Article 43, SCA 2021. DOI [10.1145/3480143](https://doi.org/10.1145/3480143).

**What it does.** Sheen, Larionov, and Pai formulate a stable volume-preserving soft-tissue simulation and add an epidermis model to capture the higher stiffness of skinned bodies. The method aims for precise volume preservation without locking artifacts and makes resolution-consistent simulation possible because the preservation term is not tied directly to mesh discretization. It is a numerical tissue solver with a skin layer, not a symbolic generator of anatomy.

**Relevance under the kernel constraints.** This is the right physical law to expose as a bounded optional stage: rest volume, material policy, activation, and skin stiffness should be explicit inputs, with volume residual and inversion checks in the result. The kernel can use the volume law to reject an impossible authored deformation or to calibrate an implicit envelope, but it should not pretend that a simulation has solved missing anatomy. LLM authors need typed controls such as `VolumePolicy` and `SkinStiffness`, not a hundred undifferentiated solver knobs. The receipt must preserve rest state, discretization, and tolerance.

### 4.3 Muscle and Fascia Simulation with Extended Position Based Dynamics

**Citation.** M. Romeo, C. Monteagudo, and D. Sánchez-Quirós. “Muscle and Fascia Simulation with Extended Position Based Dynamics.” *Computer Graphics Forum*, 39(1), 134–146, 2020. [Eurographics paper](https://diglib.eg.org/items/14e211c3-47c6-49a3-abfe-85fe94679adf).

**What it does.** This work extends position-based dynamics with constraints intended for skeletal muscle and superficial fascia. It targets the gap between realistic but expensive FEM/finite-volume methods and fast, controllable PBD, introducing constraints for muscle shape, attachment, volume, and fascia behavior. The result is a practical digital-creature simulator whose controllability is useful for visual effects, but whose output depends on iterative constraint projection and a discretized volume.

**Relevance under the kernel constraints.** Extended PBD is an attractive preview or bounded solver because its constraints are more teachable than a general constitutive model. Adopt the constraint vocabulary as a typed problem description, but require deterministic constraint ordering, fixed iteration budgets, and explicit residuals. A PBD result cannot silently become the semantic body: it is a solved layer derived from the authored anatomy graph. LLM repair should target one failed attachment or volume constraint, not ask the model to “make it more organic.” This is a good middle tier between analytic envelopes and full FEM.

### 4.4 DiffPD and differentiable muscle/tissue fitting

**Citation.** Tao Du, Kui Wu, Pingchuan Ma, Sebastien Wah, Andrew Spielberg, Daniela Rus, and Wojciech Matusik. “DiffPD: Differentiable Projective Dynamics.” *ACM Transactions on Graphics*, 40(4), 2021. [Paper](https://arxiv.org/abs/2101.05917).

**What it does.** DiffPD differentiates projective-dynamics simulation efficiently by reusing a prefactorized Cholesky decomposition. It supports penalty and complementarity contact and applies the gradients to system identification, inverse design, trajectory optimization, and control. The work makes a robust simulator more useful for fitting parameters and controls, but the differentiable path still inherits numerical, contact, and optimization concerns.

**Relevance under the kernel constraints.** DiffPD belongs in an inverse-problem interpreter: fit activation or material parameters to a declared target, then project the candidate into the symbolic vocabulary. Its contact model suggests explicit non-penetration obligations and typed complementarity residuals. It must not be the commit authority because gradient descent can find a visually acceptable but semantically invalid local minimum. For LLM authors, return the fitted parameter delta plus the residual witnesses; let the session decide whether to accept that patch. The deterministic core remains the problem constructor and validator.

### 4.5 A Second Order Cone Programming Approach for Biphasic Materials

**Citation.** Pengbin Tang, Stelian Coros, and Bernhard Thomaszewski. “A Second Order Cone Programming Approach for Simulating Biphasic Materials.” *Computer Graphics Forum*, 41(8), 87–93, 2022. [Paper](https://crl.ethz.ch/papers/Tang22SOCP.pdf).

**What it does.** Tang, Coros, and Thomaszewski formulate a solver for biphasic materials using second-order cone programming. The convex-optimization structure is aimed at robust material behavior and contact-like constraints in deformable simulation. It is not a creature-specific layer generator, but it is relevant to tissue models where a solid phase and a fluid-like or pressure-constrained phase must share a volume.

**Relevance under the kernel constraints.** The adoption is not “simulate biphasic flesh by default.” It is to recognize convex subproblems and typed feasibility certificates when a tissue law demands them. A symbolic kernel can expose a finite biphasic policy, construct a conic problem, and return either a solution with residuals or `BiphasicFeasibilityObstruction`. This is more compatible with typed verdicts than an opaque iterative fallback. LLM authors can choose a named policy and inspect an infeasible constraint set; they should not author cone matrices or infer that a failed solve means “increase softness.”

### 4.6 Skeletal-Driven Animation of Anatomical Humans via Neural Deformation Gradients

**Citation.** Gerrit Nolte, Fabian Kemper, Ulrich Schwanecke, and Mario Botsch. “Skeletal-Driven Animation of Anatomical Humans via Neural Deformation Gradients.” *Computer Graphics Forum*, 45(2), 2026. [Paper and project record](https://lamarr-institute.org/publication/skeletal-driven-animation-of-anatomical-humans-via-neural-deformation-gradients/).

**What it does.** NDG trains a neural network on high-quality FEM animation to predict deformation gradients for volumetric muscle and fat layers, rather than directly predicting vertex displacements. A matrix-exponential output enforces positive determinants, and the method reports interactive animation with fewer inversion and volume artifacts than vertex-based neural baselines. It is a data-driven surrogate for expensive simulation: fast at inference, but tied to its training distribution and unable to change material parameters without retraining.

**Relevance under the kernel constraints.** This is the modern neural continuation of the volume-preserving flesh line associated with Theodore Kim’s work, and it clarifies the boundary rather than erasing it. The transferable invariant is deformation-gradient validity, especially positive determinant and volume residual; the learned network remains a derived runtime view. A deterministic kernel can use the same gradient checks on an analytic or numerically solved layer and can optionally compare an NDG-style surrogate against the authoritative result. For LLM authors, “preserve volume and reject inverted cells” is typed and teachable; “trust the network’s anatomy” is neither. Keep training corpus, model version, and inference settings in the receipt.

### 4.7 Houdini Muscles & Tissue and the Otis direction

**Citation.** SideFX. “Introduction to Muscles and Tissues,” “Vellum Muscles and Tissue Overview,” and “Otis Muscle and Tissue Simulation.” *Houdini 20/20.5/21 Documentation*, 2023–2026. [Official documentation](https://www.sidefx.com/docs/houdini/muscles/overview.html).

**What it does.** Houdini’s documented workflows form muscles, a tissue core/fat layer, and skin from tetrahedral or surface inputs, with attachments, fiber scaling, volume preservation, sliding, caches, and post-processing. The older Vellum workflow uses separate muscle, tissue, and skin passes. The newer Otis workflow uses a VBD-based, GPU-accelerated unified muscle/tissue solve and a later skin-deform stage. SideFX explicitly describes the layer relationships and the boundary between simulated geometry and renderable skin.

**Relevance under the kernel constraints.** This is production evidence for an inside-out layered architecture, not a reason to import SOP networks. Adopt the layer graph: bone -> muscle -> tissue/fascia -> skin, with each edge carrying attachments, containment, and cache identity. A deterministic kernel can form analytic or implicit candidate layers first and hand a sealed problem to an Otis-like effect interpreter later. For LLM authors, named passes and layer-specific obstructions are far more steerable than one global “surface quality” score. The current Houdini docs also show that solver generations change; keep the semantic layer contract independent of any vendor solver.

### 4.8 Ziva RT and the post-discontinuation production landscape

**Citation.** Unity. “An update about Ziva.” *Unity News*, 2 April 2024; Ziva Dynamics. “Ziva Real-Time 2.0,” 2021–2023 product documentation and release materials. [Unity announcement](https://unity.com/blog/news/update-about-ziva).

**What it does.** Ziva VFX and Ziva Real-Time represented the commercial production path from bone and muscle simulation to real-time or baked deformation. Unity’s 2024 announcement states that it stopped actively selling and supporting Ziva VFX, Ziva RT, Ziva Face Trainer, and related products, while existing subscribers could convert licenses to a five-year term. Public evidence after the discontinuation points to studios retaining proprietary pipelines or moving toward Houdini’s muscle/tissue workflows, machine-learned deformation, and in-house systems; there is no single public replacement that inherits Ziva’s exact market position.

**Relevance under the kernel constraints.** The architectural lesson is sobering: a powerful solver product is not a durable semantic boundary. Keep a vendor-neutral solved-layer contract and treat Ziva/Houdini/ML Deformer as interpreters or calibration references. For deterministic formation, adopt explicit rest-state, attachment, volume, and skin-slide fields; do not adopt proprietary scene state. LLM authors need stable vocabulary across solver replacements, with an obstruction when a backend cannot honor a law. Ziva’s discontinuation strengthens the case for a small symbolic core whose derived solver views can be regenerated rather than stored as authority.

## 5. Skinning, envelopes, automatic rigging, and the bar to meet

### 5.1 RigNet: automatic skeleton and skinning prediction

**Citation.** Zhan Xu, Yang Zhou, Evangelos Kalogerakis, Chris Landreth, and Karan Singh. “RigNet: Neural Rigging for Articulated Characters.” *ACM Transactions on Graphics*, 39(4), Article 58, SIGGRAPH 2020. DOI [10.1145/3386569.3392379](https://doi.org/10.1145/3386569.3392379).

**What it does.** RigNet predicts an animation skeleton from an input character mesh, including joint placement and topology, and predicts skinning weights. It is an end-to-end learned system aimed at reducing manual rigging effort for articulated characters. Its generality comes from training data and geometric features, not from a closed law over every possible object.

**Relevance under the kernel constraints.** RigNet establishes the minimum practical bar: a surface system must preserve a coherent correspondence to an articulation structure, not merely produce a plausible rest mesh. A deterministic alternative should derive skeleton, part identity, and skin ownership from the same symbolic graph, with no need to infer them post hoc. Learned rigging can be used as a proposal or benchmark, but its predictions require validation for joint count, connectivity, weight locality, and symmetry. LLM authors benefit from authoring the graph directly and receiving a named obstruction instead of reverse-engineering a neural prediction.

### 5.2 SkinCells: sparse skinning with Voronoi cells

**Citation.** Egor Larionov, Igor Santesteban, Hsiao-yu Chen, Gene Lin, Philipp Herholz, Ryan Goldade, Ladislav Kavan, Doug Roble, and Tuur Stuyck. “SkinCells: Sparse Skinning using Voronoi Cells.” *Computer Graphics Forum*, 2026. [Paper](https://arxiv.org/abs/2506.14714).

**What it does.** SkinCells represents skinning weights as spatially parameterized Voronoi-cell fields with explicit sparsity control. It optimizes cell parameters over sampled poses to minimize deformation, locality, and sparsity losses, and reports robustness where biharmonic weights fail. The method supports transferring a continuous weight field to different levels of detail and related meshes, but its optimized parameters come from a PyTorch/Adam fitting loop.

**Relevance under the kernel constraints.** The deterministic residue is excellent: a sparse, spatially defined weight field with a declared maximum number of influences is easier to inspect and validate than arbitrary per-vertex weights. Adopt the cell family and sparsity law as a symbolic skinning vocabulary; replace Adam with a deterministic fit or a bounded proposal interpreter. LLM authors can name a joint field, influence budget, and locality policy, then repair a specific overlap or ambiguity. The neural optimization remains a derived calibration step. Reject unbounded influence, undefined cell ties, or skin ownership ambiguity rather than smoothing it away.

### 5.3 MagicArticulate: articulation-ready models

**Citation.** Chaoyue Song, Jianfeng Zhang, Xiu Li, Fan Yang, Yiwen Chen, Zhongcong Xu, Jun Hao Liew, Xiaoyang Guo, Fayao Liu, Jiashi Feng, and Guosheng Lin. “MagicArticulate: Make Your 3D Models Articulation-Ready.” *CVPR*, 2025, pp. 15998–16007. [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/html/Song_MagicArticulate_Make_Your_3D_Models_Articulation-Ready_CVPR_2025_paper.html).

**What it does.** MagicArticulate introduces Articulation-XL, a benchmark of more than 33,000 annotated models, predicts skeletons with an autoregressive sequence model, and predicts skinning weights with a functional diffusion process using volumetric geodesic priors. It targets diverse static meshes and converts them into assets suitable for animation. The approach is data-driven and category-general, not a symbolic body-plan compiler.

**Relevance under the kernel constraints.** This is evidence that automatic rigging now expects semantic skeletons, spatial priors, and usable deformation, not merely nearest-bone weights. The kernel should meet that bar with explicit attachment sites, part ownership, and typed correspondence policies. A learned model may propose a rig for an authored mesh, but acceptance must check graph compatibility and weight-field regularity. For LLM authors, a finite articulation vocabulary with geodesic or field-based diagnostics is teachable; a diffusion sampler is not a deterministic source of truth. Keep the predicted rig as a comparison view and report disagreements explicitly.

### 5.4 Anymate: scale of the learned rigging corpus

**Citation.** Yufan Deng, Yuhao Zhang, Chen Geng, Shangzhe Wu, and Jiajun Wu. “Anymate: A Dataset and Baselines for Learning 3D Object Rigging.” *SIGGRAPH Conference Papers ’25*, 2025, 10 pages. DOI [10.1145/3721238.3730743](https://doi.org/10.1145/3721238.3730743).

**What it does.** Anymate provides 230,000 3D assets paired with expert rigging and skinning information and evaluates sequential prediction of joints, connectivity, and skinning weights. The authors report a dataset roughly 70 times larger than earlier rigging datasets and establish learned baselines across object categories. Its contribution is primarily coverage and a reproducible benchmark for automated rigging, not a new law of surface formation.

**Relevance under the kernel constraints.** Anymate raises the empirical bar for a deterministic alternative: it must be evaluated on diverse parts, unusual proportions, and non-human articulation, not only a hand-built creature demo. The dataset can serve as an external benchmark or calibration corpus, but importing its labels would not make the runtime deterministic. The kernel should expose the same three separable obligations—joint placement, graph connectivity, and weight ownership—with independent verdicts. LLM authors can repair one obligation at a time, which is vastly more reliable than revising a monolithic rig description after a visual failure.

### 5.5 Make-It-Animatable: fast end-to-end animation readiness

**Citation.** Zhiyang Guo, Jinxu Xiang, Kai Ma, Wengang Zhou, Houqiang Li, and Ran Zhang. “Make-It-Animatable: An Efficient Framework for Authoring Animation-Ready 3D Characters.” *CVPR*, 2025, pp. 10783–10792. [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_Make-It-Animatable_An_Efficient_Framework_for_Authoring_Animation-Ready_3D_Characters_CVPR_2025_paper.html).

**What it does.** Make-It-Animatable addresses the post-generation pipeline by producing an animation-ready asset with a skeleton and skinning, including support for mesh and 3D Gaussian inputs. Its comparison table makes the production expectation explicit: template-free inputs, alterable skeletons, pose-to-rest support, hand animation, and sub-second or near-sub-second rigging are now meaningful criteria. The method is optimized for learned generalization and speed rather than formal semantic guarantees.

**Relevance under the kernel constraints.** This is a bar-setting paper, not a direct implementation candidate. A deterministic kernel should compete on stable correspondences, reproducible output, and typed failure rather than raw throughput alone. The useful adoption is a qualification matrix: source representation, skeleton authority, alterability, rest-pose mapping, deformation quality, and failure coverage. LLM authors should receive that matrix as structured feedback. If a body cannot be made articulation-ready under the authored contract, return the exact obstruction; a fast but invalid rig is merely a faster way to produce downstream entropy.

## 6. Explicit 2026 verdict

### Are gradient-based implicit blends and implicit skinning superseded?

**No—not for a deterministic symbolic kernel.** They are not the overall frontier for learned reconstruction, inverse fitting, or data-driven animation, but they remain lawful and useful components when their scope is narrowed. The 2020–2026 literature does not replace the core implicit ideas with one universal representation. Instead, it adds better bounds, differentiable fitting, neural correspondences, sparse fields, and production solvers around them.

The 2013-era gradient-based blend idea remains relevant because the kernel’s problem is local compatibility between authored fields. Its lawful successor is not “more neural”: it is a bounded, gradient-aware, Lipschitz-accounted composition operator with an explicit support region and a typed failure when calibration is unavailable. LipMLP and Lipschitz Pruning strengthen this direction with slope bounds and regional equivalence. DiffCSG and DeepMesh show how to fit or render symbolic fields differentiably, but they do not erase the need for a deterministic evaluator.

Implicit skinning likewise remains a strong derived view when the base field, correspondence, and offset policy are explicit. SNARF demonstrates the neural side’s ability to learn continuous correspondence; SkinCells demonstrates a useful spatial field and sparsity control; MagicArticulate and Anymate demonstrate the quality and coverage bar for automatic rigging. None provides a better semantic authority for an authored creature than a typed skeleton/part graph whose skinning field can be recomputed and checked. Neural work occupies the data-driven side: proposal, fitting, calibration, reconstruction, or acceleration.

### Adoption ranking for 2026+

1. **Adopt immediately: typed procedural genome and part graph.** Use Infinigen, Infinigen-Articulated, AWOL, RAW, DeepCAD, and Text2CAD as evidence for a finite body-plan language: named parts, stable IDs, profiles, attachments, joints, layer roles, and a canonical seed. The LLM authors this program; it does not author the final mesh.
2. **Adopt immediately: local bounded composition with field certificates.** Replace global smooth-min semantics with contact-local operators. Carry exact/conservative status, Lipschitz bounds, support regions, gradient calibration, and topology witnesses. Lipschitz Pruning is the highest-leverage performance and proof idea; DiffCSG is an offline fitting aid.
3. **Adopt immediately: transactional compiler and visual feedback loop.** Use the SceneCraft/CADCodeVerify pattern: plan, compile, render, inspect, issue typed diagnostics, and apply a scoped patch. Compiler facts outrank VLM judgements. A render can suggest a repair; only descent and gluing can authorize it.
4. **Adopt next: analytic or bounded quasi-static tissue layers.** Start with deterministic muscle envelopes, volume-preserving constraints, explicit attachments, and a skin offset/relaxation stage. Use extended PBD or a small fixed solver as an interpreter, with stable ordering and residual receipts. Do not begin with a general FEM or a vendor scene graph.
5. **Adopt next: sparse field-based skinning.** A SkinCells-like spatial weight family with a closed influence budget is preferable to arbitrary per-vertex weights. It gives a compact authoring surface, stable LOD transfer, and meaningful obstruction types. Neural rigging remains a benchmark and proposal source.
6. **Adopt conditionally: numerical and differentiable solvers.** EMU, DiffPD, biphasic cone programming, Houdini Otis, or an equivalent backend can improve derived deformation quality. They require effect-boundary isolation, sealed versions/settings, deterministic input serialization, and explicit non-convergence/inversion verdicts. They must never silently rewrite the authored body.
7. **Do not adopt as authority: neural implicit fields, diffusion rigs, arbitrary Blender Python, or image-only verification.** They are useful views and proposal engines. They are not deterministic, typed, or sufficient to establish topology, attachment, thickness, or law satisfaction.

The 2026 architecture should therefore be hybrid but not vague: **symbolic genome and typed part graph as authority; bounded implicit fields as the surface algebra; local solver interpreters for tissue; sparse field skinning for correspondence; LLM/VLM loops for proposal and repair; rendered meshes, neural fits, and vendor simulations as derived views.** The boundary is the product. Anything that cannot cross it with a typed witness remains an obstruction, however beautiful its screenshot.
