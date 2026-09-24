param(
    # Credential-manager account name, e.g. TYPESAFE_API_KEY. The credential
    # is read from the "POK-Ai" service created by the Tauri app
    # (apps/desktop/src-tauri/src/lib.rs persist_provider_key). Prints the
    # secret to stdout; redirect to a variable, never print it to a log.
    [Parameter(Mandatory = $true)][string]$Name,
    # Print only the secret's length instead of the secret itself.
    [switch]$LengthOnly
)

$ErrorActionPreference = "Stop"

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class PokCredential {
  [StructLayout(LayoutKind.Sequential)]
  public struct CREDENTIAL {
    public uint Flags;
    public int Type;
    public IntPtr TargetName;
    public IntPtr Comment;
    public System.Runtime.InteropServices.ComTypes.FILETIME LastWritten;
    public uint CredentialBlobSize;
    public IntPtr CredentialBlob;
    public uint Persist;
    public uint AttributeCount;
    public IntPtr Attributes;
    public IntPtr TargetAlias;
    public IntPtr UserName;
  }
  [DllImport("advapi32.dll", SetLastError = true)]
  public static extern bool CredRead(string target, int type, int reservedFlag, out IntPtr credentialPtr);
  [DllImport("advapi32.dll")]
  public static extern void CredFree(IntPtr buffer);
}
"@

$target = "$Name.POK-Ai"
$ptr = [IntPtr]::Zero
if (-not [PokCredential]::CredRead($target, 1, 0, [ref]$ptr)) {
    Write-Error "credential '$target' was not found (win32 error $([System.Runtime.InteropServices.Marshal]::GetLastWin32Error()))"
    exit 1
}
try {
    $cred = [System.Runtime.InteropServices.Marshal]::PtrToStructure($ptr, [type][PokCredential+CREDENTIAL])
    $bytes = New-Object byte[] $cred.CredentialBlobSize
    [System.Runtime.InteropServices.Marshal]::Copy($cred.CredentialBlob, $bytes, 0, $cred.CredentialBlobSize)
    $secret = [System.Text.Encoding]::Unicode.GetString($bytes)
    if ($LengthOnly) {
        Write-Output ("SECRET_LENGTH=" + $secret.Length)
    } else {
        Write-Output $secret
    }
} finally {
    [PokCredential]::CredFree($ptr) | Out-Null
}