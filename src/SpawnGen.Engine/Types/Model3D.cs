using SpawnGen.Engine.Nodes;

namespace SpawnGen.Engine.Types;

/// <summary>A 3D model moving between nodes: the bytes of a file (a .glb, glTF binary, with its textures inside) and its extension.</summary>
[Socket("FILE_3D")]
public sealed record Model3D(byte[] Bytes, string Extension = "glb");
