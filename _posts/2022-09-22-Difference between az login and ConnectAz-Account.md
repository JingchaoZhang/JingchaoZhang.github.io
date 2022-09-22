---
layout: single
author_profile: false
---

After doing `az login` and `az account set -s XXX`, I still got the following error message:
```bash
Subscription (xxx) or Tenant (xxx) doesn't exists.
```

The solution is to login using `Connect-AzAccount` instead of `az login`. Then change subscription use `Set-AzContext xxx`.

You can verify the setting is correct using the command below:
```bash
Select-AzSubscription -SubscriptionId xxx -TenantId xxx
```

So, what is the different between `az login` and `Connect-AzAccount`? Here is a [GitHub thread](https://github.com/MicrosoftDocs/azure-docs-powershell/issues/1663) on this topic.

In short, `az login` and `Connect-AzAccount` are aliases. But, in order to use cmdlets from the Az PowerShell modules, you will need to use `Connect-AzAccount` instead of `az login`.
